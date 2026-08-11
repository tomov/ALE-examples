"""fmri_play.py -- run ANY Arcade Learning Environment game as an fMRI task.

A small, generic proof-of-concept framework that adapts ALE/Atari games for
neuroimaging experiments. It provides the pieces every scanner task needs:

  * a uniform, fixed-size display -- every game (all render 210x160) is
    aspect-fit and centered with black padding so screen geometry is constant;
  * a fixation cross ("+") shown until the scanner trigger ("=") arrives, which
    anchors every timestamp in the session (t0);
  * a JSON-defined *curriculum* of phases -- fixation / message / game / survey
    -- that plays in sequence, with inter-block intervals (IBIs);
  * two ways to end a game block: play one/several episodes, or replay a level
    on repeat for a fixed duration (decouples data-per-condition from skill);
  * compact, fully-replayable logging: per frame we save action, reward,
    terminal, lives, RAM (128B internal state) and timestamps, plus each
    episode's exact ALE state (clone_state) + seed. Because ALE is
    deterministic, this reconstructs every pixel frame offline.

Usage:
    python fmri_play.py --subject sub-01                 # built-in demo curriculum
    python fmri_play.py --subject sub-01 --curriculum my_curriculum.json
    python fmri_play.py --subject sub-01 --dummy-trigger # auto-fire trigger (testing)

Design mirrors the fMRI/MEG adaptations in ~/Documents/projects/DBP/{vgdl,mario_task}
but stripped to a POC and made game-agnostic. See build_demo_curriculum() for the
curriculum schema.
"""

import argparse
import json
import os
import pickle
import sys
import time

import ale_py
import gymnasium as gym
import numpy as np
import pygame

gym.register_envs(ale_py)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TRIGGER_KEY = "="          # scanner trigger character (as in the vgdl fMRI task)
EXPERIMENTER_KEY = " "     # experimenter advances the pre-scan instruction screen
ALE_FRAME_SIZE = (160, 210)  # (w, h) -- every ALE game renders at this resolution
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (220, 220, 220)
FIX_COLOR = (255, 255, 255)

# Deterministic play: 1 game-frame per input, no "sticky actions".
ENV_KWARGS = dict(frameskip=1, repeat_action_probability=0.0)


# --------------------------------------------------------------------------- #
# Keyboard -> action mapping (built per game from its own action meanings)
# --------------------------------------------------------------------------- #
DIRECTION_KEYS = {
    "UP": (pygame.K_UP,),
    "DOWN": (pygame.K_DOWN,),
    "LEFT": (pygame.K_LEFT,),
    "RIGHT": (pygame.K_RIGHT,),
    "UPRIGHT": (pygame.K_UP, pygame.K_RIGHT),
    "UPLEFT": (pygame.K_UP, pygame.K_LEFT),
    "DOWNRIGHT": (pygame.K_DOWN, pygame.K_RIGHT),
    "DOWNLEFT": (pygame.K_DOWN, pygame.K_LEFT),
}


def build_keymap(env):
    """Return {frozenset(pygame_keys): action_index} from a game's action set."""
    keymap = {}
    for action, meaning in enumerate(env.unwrapped.get_action_meanings()):
        if meaning == "NOOP":
            continue
        fire = meaning.endswith("FIRE")
        direction = meaning[:-4] if fire and meaning != "FIRE" else meaning
        keys = DIRECTION_KEYS.get(direction, ())
        if fire:
            keys = keys + (pygame.K_SPACE,)
        if keys:
            keymap[frozenset(keys)] = action
    return keymap


def action_from_keys(pressed, keymap, noop=0):
    """Pick the action whose key-combo best matches currently-pressed keys.

    Prefer the most specific match (largest key set that is fully held), so
    e.g. RIGHT+FIRE wins over RIGHT alone when both space and right are down.
    """
    best_action, best_len = noop, 0
    for keys, action in keymap.items():
        if keys <= pressed and len(keys) > best_len:
            best_action, best_len = action, len(keys)
    return best_action


# --------------------------------------------------------------------------- #
# Display: fixed-size window, centered aspect-fit game frame, text & fixation
# --------------------------------------------------------------------------- #
class Display:
    def __init__(self, size=(800, 600), fullscreen=False):
        pygame.init()
        pygame.mouse.set_visible(False)
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("fmri_play (ALE)")
        self.size = self.screen.get_size()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 32)
        self.fix_font = pygame.font.Font(pygame.font.get_default_font(), 80)

        # Precompute the centered, aspect-preserving blit rect for game frames.
        fw, fh = ALE_FRAME_SIZE
        scale = min(self.size[0] / fw, self.size[1] / fh)
        self.game_rect = pygame.Rect(0, 0, int(fw * scale), int(fh * scale))
        self.game_rect.center = (self.size[0] // 2, self.size[1] // 2)

    def draw_frame(self, obs):
        """Blit an ALE RGB observation (H,W,3), scaled & centered with padding."""
        self.screen.fill(BG_COLOR)
        surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))  # -> (W,H)
        surf = pygame.transform.scale(surf, self.game_rect.size)
        self.screen.blit(surf, self.game_rect.topleft)
        pygame.display.flip()

    def draw_text(self, text, color=TEXT_COLOR, font=None):
        font = font or self.font
        self.screen.fill(BG_COLOR)
        lines = text.split("\n")
        total_h = sum(font.size(ln)[1] for ln in lines)
        y = (self.size[1] - total_h) // 2
        for ln in lines:
            surf = font.render(ln, True, color)
            rect = surf.get_rect(center=(self.size[0] // 2, y + surf.get_height() // 2))
            self.screen.blit(surf, rect)
            y += surf.get_height()
        pygame.display.flip()

    def draw_fixation(self):
        self.draw_text("+", color=FIX_COLOR, font=self.fix_font)

    def close(self):
        pygame.quit()


# --------------------------------------------------------------------------- #
# Session clock & logging
# --------------------------------------------------------------------------- #
class Clock:
    """Anchored at the scanner trigger; provides trigger-relative + epoch time."""

    def __init__(self):
        self.t0_perf = None
        self.t0_epoch = None

    def anchor(self):
        self.t0_perf = time.perf_counter()
        self.t0_epoch = time.time()

    def rel(self):
        return time.perf_counter() - self.t0_perf

    def epoch(self):
        return time.time()


class Logger:
    """Writes per-game-block .npz files and a session manifest.json."""

    def __init__(self, outdir, subject, curriculum, clock):
        self.outdir = outdir
        self.clock = clock
        os.makedirs(outdir, exist_ok=True)
        self.manifest = {
            "subject": subject,
            "curriculum": curriculum,
            "start_epoch": None,   # filled at trigger
            "phases": [],          # one entry per phase, in order
        }

    def set_trigger_time(self):
        self.manifest["start_epoch"] = self.clock.t0_epoch
        self.manifest["trigger_perf"] = self.clock.t0_perf

    def log_phase(self, entry):
        self.manifest["phases"].append(entry)

    def save_game_block(self, block_index, game, frames):
        """frames: dict of parallel lists collected during a game block."""
        path = os.path.join(self.outdir, f"block-{block_index:02d}_{game}.npz")
        arrays = dict(
            actions=np.array(frames["action"], dtype=np.int16),
            rewards=np.array(frames["reward"], dtype=np.float32),
            terminal=np.array(frames["terminal"], dtype=bool),
            lives=np.array(frames["lives"], dtype=np.int16),
            episode_id=np.array(frames["episode_id"], dtype=np.int32),
            t_rel=np.array(frames["t_rel"], dtype=np.float64),
            t_epoch=np.array(frames["t_epoch"], dtype=np.float64),
            ram=np.array(frames["ram"], dtype=np.uint8),
            # Exact ALE machine state (pickled clone_state) for EVERY frame, so
            # each frame is independently restorable to a bit-exact screen with
            # a single step() -- no reliance on deterministic action replay.
            states=np.array(frames["states"], dtype=object),
            # One clone_state + seed per episode start (redundant with `states`
            # but convenient, and the anchor for action-replay reconstruction).
            init_states=np.array(frames["init_states"], dtype=object),
            episode_seeds=np.array(frames["episode_seeds"], dtype=np.int64),
            game=game,
        )
        # Optional lossless pixels: store the indexed (palettized) screen +
        # palette. RGB == palette[screen_index] exactly, so this is lossless;
        # zlib in savez_compressed exploits the large flat regions (~0.1KB/frame
        # vs ~100KB/frame for raw RGB). Only present if save_pixels was enabled.
        if frames["screen_index"]:
            arrays["screen_index"] = np.array(frames["screen_index"], dtype=np.uint8)
            arrays["palette"] = np.array(frames["palette"], dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return path

    def save_manifest(self):
        path = os.path.join(self.outdir, "manifest.json")
        with open(path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return path


# --------------------------------------------------------------------------- #
# Low-level event / pacing helpers
# --------------------------------------------------------------------------- #
def check_quit():
    """Poll for a hard quit (window close or ESC). Returns True if quitting."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def wait_for_key(display, unicode_char, dummy=False):
    """Block until `unicode_char` is pressed (or immediately if dummy)."""
    if dummy:
        time.sleep(0.1)
        return
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                if event.unicode == unicode_char:
                    return
        time.sleep(0.005)


# --------------------------------------------------------------------------- #
# Phase handlers
# --------------------------------------------------------------------------- #
def run_fixation(display, clock, logger, phase, index):
    """Show '+' for a fixed duration (used for pre-run and IBIs)."""
    duration = phase.get("duration", 2.0)
    onset = clock.rel()
    display.draw_fixation()
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        if check_quit():
            raise KeyboardInterrupt
        time.sleep(0.005)
    logger.log_phase({
        "index": index, "type": "fixation",
        "onset": onset, "offset": clock.rel(), "duration": duration,
    })


def run_message(display, clock, logger, phase, index):
    """Show a text message for a duration, or until a key is pressed."""
    text = phase.get("text", "")
    duration = phase.get("duration")  # None => wait for key
    onset = clock.rel()
    display.draw_text(text)
    if duration is None:
        wait_for_key(display, phase.get("key", " "))
    else:
        end = time.perf_counter() + duration
        while time.perf_counter() < end:
            if check_quit():
                raise KeyboardInterrupt
            time.sleep(0.005)
    logger.log_phase({
        "index": index, "type": "message", "text": text,
        "onset": onset, "offset": clock.rel(),
    })


def run_game(display, clock, logger, phase, index):
    """Play one ALE game block and log every frame.

    Phase fields:
        game        : ALE env id, e.g. "ALE/Pong-v5"
        mode        : "episode" (play until game over) or "duration" (replay
                      until `duration` seconds elapse). Default "duration".
        duration    : seconds, for mode == "duration" (default 30)
        n_episodes  : episodes to play, for mode == "episode" (default 1)
        max_duration: hard safety cap in seconds for mode == "episode", so a
                      never-ending episode can't stall the session (default 300)
        fps         : target game frames per second (default 30)
        seed        : base RNG seed (default derived from block index)
        save_pixels : if true, ALSO store the lossless indexed screen per frame.
                      OFF by default -- see the warning below. (default False)
    """
    game = phase["game"]
    mode = phase.get("mode", "duration")
    duration = phase.get("duration", 30.0)
    n_episodes = phase.get("n_episodes", 1)
    fps = phase.get("fps", 30)
    base_seed = phase.get("seed", 1000 + index)
    save_pixels = phase.get("save_pixels", False)
    dt = 1.0 / fps
    # Hard wall-clock cap so the block always terminates: `duration` in
    # duration mode, else a generous safety limit in episode mode.
    cap = duration if mode == "duration" else phase.get("max_duration", 300.0)

    if save_pixels:
        # Per-frame full state already makes every frame reconstructable
        # losslessly and cheaply; saving raw pixels on top is rarely necessary.
        print("\n" + "!" * 70, file=sys.stderr)
        print("!! WARNING: save_pixels is ON for block %d (%s)." % (index, game),
              file=sys.stderr)
        print("!! This stores the screen for EVERY frame. Even losslessly "
              "compressed", file=sys.stderr)
        print("!! this is far larger than the state log, and unnecessary: the "
              "per-frame", file=sys.stderr)
        print("!! clone_state already reconstructs every pixel exactly. Only "
              "enable this", file=sys.stderr)
        print("!! if a downstream tool truly cannot replay states offline.",
              file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)

    env = gym.make(game, render_mode="rgb_array", **ENV_KWARGS)
    keymap = build_keymap(env)

    frames = {k: [] for k in (
        "action", "reward", "terminal", "lives", "episode_id",
        "t_rel", "t_epoch", "ram", "states")}
    frames["init_states"] = []
    frames["episode_seeds"] = []
    frames["screen_index"] = []   # lossless indexed pixels (only if save_pixels)
    palette = np.zeros((256, 3), dtype=np.uint8)  # index -> RGB, block-wide
    palette_seen = np.zeros(256, dtype=bool)

    onset = clock.rel()
    block_end = time.perf_counter() + cap
    episode_id = 0
    total_reward = 0.0
    user_quit = False  # ESC / window close, as opposed to normal time/episode end

    while not user_quit and time.perf_counter() < block_end:
        seed = base_seed + episode_id
        obs, info = env.reset(seed=seed)
        # Save the exact starting state + seed so this episode is replayable.
        frames["init_states"].append(pickle.dumps(
            env.unwrapped.clone_state(include_rng=True)))
        frames["episode_seeds"].append(seed)

        terminated = truncated = False
        next_t = time.perf_counter()
        while not (terminated or truncated):
            # --- pacing: hold to target fps ---
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

            if check_quit():
                user_quit = True
                break

            pressed = frozenset(
                k for k in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                            pygame.K_RIGHT, pygame.K_SPACE)
                if pygame.key.get_pressed()[k])
            action = action_from_keys(pressed, keymap)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ale = env.unwrapped.ale

            frames["action"].append(action)
            frames["reward"].append(reward)
            frames["terminal"].append(bool(terminated or truncated))
            frames["lives"].append(info.get("lives", 0))
            frames["episode_id"].append(episode_id)
            frames["t_rel"].append(clock.rel())
            frames["t_epoch"].append(clock.epoch())
            frames["ram"].append(ale.getRAM().copy())
            # Exact machine state for this frame -> independently restorable.
            frames["states"].append(pickle.dumps(
                env.unwrapped.clone_state(include_rng=True)))

            if save_pixels:
                idx = ale.getScreen()          # (210,160) uint8 palette indices
                frames["screen_index"].append(idx.copy())
                # Learn any new index->RGB mappings for this block's palette.
                new = np.unique(idx)
                new = new[~palette_seen[new]]
                if new.size:
                    flat_i = idx.reshape(-1)
                    flat_c = obs.reshape(-1, 3)
                    for i in new:
                        palette[i] = flat_c[flat_i == i][0]
                        palette_seen[i] = True

            display.draw_frame(obs)

            # Stop mid-episode once the block's wall-clock cap is reached.
            if time.perf_counter() >= block_end:
                break

        episode_id += 1
        if mode == "episode" and episode_id >= n_episodes:
            break

    env.close()
    frames["palette"] = palette  # block-wide index->RGB (only used if save_pixels)
    path = logger.save_game_block(index, game.split("/")[-1], frames)
    logger.log_phase({
        "index": index, "type": "game", "game": game, "mode": mode,
        "onset": onset, "offset": clock.rel(),
        "n_episodes": episode_id, "n_frames": len(frames["action"]),
        "total_reward": total_reward, "data_file": os.path.basename(path),
    })
    if user_quit:  # ESC / window close => end the whole session, saving data
        raise KeyboardInterrupt


def run_survey(display, clock, logger, phase, index):
    """Minimal 1-7 Likert survey. LEFT/RIGHT change rating, ENTER confirms."""
    questions = phase.get("questions", [])
    n_points = phase.get("n_points", 7)
    onset = clock.rel()
    responses = []
    for q in questions:
        value = (n_points + 1) // 2  # start in the middle
        confirmed = False
        while not confirmed:
            scale = "  ".join(
                (f"[{i}]" if i == value else f" {i} ") for i in range(1, n_points + 1))
            display.draw_text(f"{q}\n\nDisagree      Agree\n{scale}\n\n"
                              "(LEFT/RIGHT to rate, ENTER to confirm)")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_LEFT:
                        value = max(1, value - 1)
                    elif event.key == pygame.K_RIGHT:
                        value = min(n_points, value + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        confirmed = True
            time.sleep(0.005)
        responses.append({"question": q, "value": value, "t_rel": clock.rel()})
    logger.log_phase({
        "index": index, "type": "survey",
        "onset": onset, "offset": clock.rel(), "responses": responses,
    })


PHASE_HANDLERS = {
    "fixation": run_fixation,
    "message": run_message,
    "game": run_game,
    "survey": run_survey,
}


# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #
def build_demo_curriculum():
    """A small built-in curriculum demonstrating every phase type.

    A curriculum is just an ordered list of phase dicts. Games are separated by
    a game-name cue (message) and a fixation IBI, and the session ends with a
    short survey -- the same run->cue->fixation->game->...->survey shape used in
    the vgdl/mario fMRI tasks, minimized for a POC.
    """
    games = ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5"]
    curriculum = []
    for game in games:
        name = game.split("/")[-1].replace("-v5", "")
        curriculum += [
            {"type": "message", "text": name, "duration": 2.0},
            {"type": "fixation", "duration": 2.0},
            {"type": "game", "game": game, "mode": "duration",
             "duration": 30.0, "fps": 30},
        ]
    curriculum.append({"type": "fixation", "duration": 4.0})  # post-run HRF settle
    curriculum.append({"type": "survey", "questions": [
        "I was fully absorbed in the games.",
        "The games were too difficult.",
        "I feel tired.",
    ]})
    return curriculum


def load_curriculum(path):
    with open(path) as f:
        data = json.load(f)
    # Accept either a bare list or {"curriculum": [...]}.
    return data["curriculum"] if isinstance(data, dict) else data


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run ALE games as an fMRI task.")
    parser.add_argument("--subject", default="sub-test", help="subject id")
    parser.add_argument("--curriculum", help="path to a curriculum JSON file")
    parser.add_argument("--outdir", help="output dir (default data/<subject>_<ts>)")
    parser.add_argument("--size", default="800x600", help="window size, e.g. 800x600")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--dummy-trigger", action="store_true",
                        help="auto-fire the scanner trigger (for testing)")
    args = parser.parse_args()

    curriculum = (load_curriculum(args.curriculum) if args.curriculum
                  else build_demo_curriculum())
    w, h = (int(x) for x in args.size.lower().split("x"))
    outdir = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")

    display = Display(size=(w, h), fullscreen=args.fullscreen)
    clock = Clock()
    logger = Logger(outdir, args.subject, curriculum, clock)

    try:
        # --- pre-scan: experimenter readies subject, then wait for trigger ---
        display.draw_text("Please keep your head as still as possible.\n\n"
                          "(experimenter: press SPACE when ready)")
        wait_for_key(display, EXPERIMENTER_KEY, dummy=args.dummy_trigger)

        display.draw_text("Waiting for scanner...")
        wait_for_key(display, TRIGGER_KEY, dummy=args.dummy_trigger)

        # --- trigger received: anchor the session clock ---
        clock.anchor()
        logger.set_trigger_time()

        for index, phase in enumerate(curriculum):
            handler = PHASE_HANDLERS.get(phase["type"])
            if handler is None:
                raise ValueError(f"unknown phase type: {phase['type']!r}")
            handler(display, clock, logger, phase, index)

        display.draw_text("Done. Thank you!")
        time.sleep(2.0)
    except KeyboardInterrupt:
        print("Interrupted -- saving partial data.", file=sys.stderr)
    finally:
        manifest_path = logger.save_manifest()
        display.close()
        print(f"Saved session to: {outdir}")
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
