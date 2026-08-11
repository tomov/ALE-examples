# ALE-examples

Small standalone examples for playing and adapting
[Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
(Atari 2600) games as a **human** — including a proof-of-concept framework for
running any ALE game as an **fMRI/neuroimaging task**.

It is the Atari analogue of the fMRI/MEG game tasks in
`~/Documents/projects/DBP/vgdl` (VGDL, fMRI) and
`~/Documents/projects/DBP/mario_task` (Mario, MEG/EEG), stripped down to a
proof of concept and made game-agnostic.

These scripts depend only on the `ale-py` PyPI package (no ALE source build
required). The Atari ROMs ship inside the `ale-py` wheel.

## Contents

| File | Purpose |
|------|---------|
| `play_atari.py`   | Play any Atari game by hand with the keyboard. Useful for sanity checks. |
| `fmri_play.py`    | fMRI framework: fixation → scanner trigger (`=`) → JSON curriculum → replayable per-frame logging. |

## Install

```bash
conda create -n ale-play python=3.11
conda activate ale-play
pip install -r requirements.txt
```

> If your default pip index is a private registry, add
> `--index-url https://pypi.org/simple`.

## Quick start

```bash
# Play a game by hand (arrow keys move, SPACE fires, ESC quits):
python play_atari.py ALE/Pong-v5

# Run the fMRI demo task (auto-fires the scanner trigger for testing):
python fmri_play.py --subject sub-01 --dummy-trigger
```

---

## Do I need a full ALE source build?

**No.** `fmri_play.py` and `play_atari.py` depend only on packages installed
from PyPI — nothing from the ALE C++/build tree is imported at runtime. You can
copy the two `.py` files anywhere and run them, as long as the packages in
`requirements.txt` are installed. (Verified by running `fmri_play.py` from a
scratch directory.)

A source build of ALE is only needed if you want to **build `ale-py` from
source**, add a new ROM, or modify the emulator — none of which these scripts
require. Atari ROMs ship inside the `ale-py` wheel.

## What is the underlying game engine?

Three layers:

1. **Stella** — the actual Atari 2600 emulator (C++). This is the "game
   engine"; it runs the original ROM. You'll see `[Powered by Stella]` printed
   on startup.
2. **ALE (Arcade Learning Environment)** — a C++ layer over Stella that exposes
   frames, RAM, actions, score, and lives, plus save/restore of emulator state.
   Surfaced to Python as **`ale_py`** (via nanobind bindings), and registered as
   Gymnasium environments.
3. **Gymnasium** — the `env.reset()/step()` API we drive.

**pygame is *not* the game engine.** In `fmri_play.py` pygame is used only for
presentation: opening the display window, reading the keyboard, and blitting
the emulator's frames. `play_atari.py` similarly uses Gymnasium's
`utils.play`, which also renders via pygame. All actual game logic is Stella.

---

## Running the fMRI task

```bash
conda activate ale-play

# Built-in demo curriculum (Pong -> Breakout -> SpaceInvaders, then a survey):
python fmri_play.py --subject sub-01

# Your own curriculum:
python fmri_play.py --subject sub-01 --curriculum my_curriculum.json

# Testing without a scanner (auto-fires the trigger):
python fmri_play.py --subject sub-01 --dummy-trigger
```

Flow at runtime:

1. Experimenter screen — press **SPACE** when the subject is ready.
2. "Waiting for scanner..." — the task blocks until the scanner **trigger
   (`=`)** arrives. This instant anchors the session clock (`t0`); every logged
   timestamp is relative to it.
3. The curriculum phases play in order.
4. `ESC` (or closing the window) ends early but still saves collected data.

Other flags: `--size 1024x768`, `--fullscreen`, `--outdir PATH`.

---

## Curriculum format

A curriculum is an ordered JSON list of **phases**. Either a bare list or
`{"curriculum": [...]}` is accepted. Phase types:

```jsonc
// Fixation cross "+" for a fixed time (pre-run settle, inter-block interval)
{"type": "fixation", "duration": 2.0}

// Text screen. With "duration": shown for that long. Without: waits for a key
// (default SPACE; set "key": "x" to change) — good for instructions.
{"type": "message", "text": "Breakout", "duration": 2.0}

// A game block.
{"type": "game",
 "game": "ALE/Pong-v5",   // any ALE env id
 "mode": "duration",       // "duration" = replay until time is up;
                           // "episode"  = play until game over
 "duration": 30.0,         // seconds (duration mode)
 "n_episodes": 1,          // episodes to play (episode mode)
 "max_duration": 300.0,    // hard wall-clock safety cap (episode mode)
 "fps": 30,                // target game frames per second
 "seed": 1234,             // base RNG seed (optional; else derived from index)
 "save_pixels": false}     // also store lossless pixels -- see warning below

// Minimal 1..N Likert survey. LEFT/RIGHT to rate, ENTER to confirm.
{"type": "survey",
 "n_points": 7,
 "questions": ["I was fully absorbed in the games.", "I feel tired."]}
```

**Two ways a game block ends** (mirroring the reference tasks):

- `mode: "duration"` — the level is replayed on repeat until `duration` seconds
  elapse (episodes restart transparently). This is the vgdl convention: it
  decouples the amount of fMRI data per condition from the subject's skill.
- `mode: "episode"` — play a fixed number of episodes (`n_episodes`) to
  completion. A `max_duration` cap guarantees the block can never hang (e.g. if
  the subject never launches the ball in Breakout).

To build a curriculum programmatically, see `build_demo_curriculum()` in
`fmri_play.py`.

---

## Uniform screen geometry

All ALE games render at **210×160**. `fmri_play.py` aspect-fits the frame and
centers it in a fixed-size window, padding the rest with black. Screen geometry
is therefore identical across every game, regardless of the individual game's
aspect ratio.

---

## Output & data format

Each session writes to `data/<subject>_<timestamp>/` (override with
`--outdir`):

- **`manifest.json`** — subject, the full curriculum, the trigger epoch time
  (`start_epoch`), and one entry per phase with `onset`/`offset` (trigger-
  relative seconds). Game phases also record frame count, episode count, total
  reward, and the data file; survey phases record responses.

- **`block-NN_<Game>.npz`** — one per game block, with parallel per-frame
  arrays:

  | key | shape | meaning |
  |-----|-------|---------|
  | `actions` | (N,) | action index taken each frame |
  | `rewards` | (N,) | reward that frame |
  | `terminal` | (N,) | episode ended this frame |
  | `lives` | (N,) | remaining lives |
  | `episode_id` | (N,) | which episode within the block |
  | `t_rel` | (N,) | seconds since scanner trigger |
  | `t_epoch` | (N,) | wall-clock epoch time |
  | `ram` | (N, 128) | Atari 2600 RAM — the compact internal game state |
  | `states` | (N,) | pickled `clone_state()` for **every** frame (exact machine state) |
  | `init_states` | (n_episodes,) | pickled `clone_state()` at each episode start |
  | `episode_seeds` | (n_episodes,) | RNG seed used per episode |
  | `screen_index` | (N, 210, 160) | *optional* — palette index per pixel (only if `save_pixels`) |
  | `palette` | (256, 3) | *optional* — index → RGB, so `palette[screen_index] == RGB` |

### Reconstruction (three independent ways)

The Atari's 128 bytes of RAM are **not** enough to rebuild the screen — the
picture is generated on the fly by the TIA video chip, and the full machine
state is ~14 KB (CPU + TIA + RIOT timers + RNG), of which RAM is <1%. So we save
the full `clone_state`, which *is* sufficient. Three ways to get frames back,
in decreasing robustness:

1. **Per-frame full state (default, determinism-free).** `states[i]` is the
   exact machine state; `restore_state(states[i])` reproduces frame `i`
   bit-exactly with no assumptions. Costs ~0.6 KB/frame compressed.
2. **Action replay.** With `frameskip=1, repeat_action_probability=0` ALE is
   deterministic, so `init_states[ep]` + the action log reproduces an episode
   frame-for-frame. (Relies on the determinism assumption.)
3. **Stored pixels (opt-in).** If `save_pixels` was set, `palette[screen_index]`
   *is* the RGB frame — no emulator needed. **Lossless.**

> ⚠️ **`save_pixels` warning.** Storing pixels every frame is unnecessary for
> reconstruction (options 1 and 2 already give bit-exact frames) and prints a
> loud warning at runtime. We store the *indexed* screen + palette rather than
> raw RGB: Atari frames use a small fixed palette with large flat regions, so
> zlib compresses this losslessly to ~0.25 KB/frame on top of the state log —
> versus ~100 KB/frame for raw RGB. Only enable it if a downstream tool truly
> cannot replay states offline.

Offline reconstruction sketch (all three ways):

```python
import numpy as np, pickle, gymnasium as gym, ale_py
gym.register_envs(ale_py)

d = np.load("block-00_Pong-v5.npz", allow_pickle=True)
env = gym.make(str(d["game"]) if "/" in str(d["game"]) else "ALE/Pong-v5",
               render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
u = env.unwrapped; env.reset()

# 1) Per-frame full state -> exact frame i, no determinism needed:
i = 10
u.restore_state(pickle.loads(d["states"][i]))
frame_i = u.ale.getScreenRGB()

# 2) Action replay for an episode:
ep = 0; mask = d["episode_id"] == ep
u.restore_state(pickle.loads(d["init_states"][ep]))
frames = [env.step(int(a))[0] for a in d["actions"][mask]]

# 3) If save_pixels was on, pixels are stored directly (lossless):
if "screen_index" in d.files:
    frame_i = d["palette"][d["screen_index"][i]]   # == getScreenRGB()
```

---

## Future work / TODO

This is a proof of concept. For a real study, the following would be worth
adding (roughly in priority order):

### Timing precision
- [ ] **Photodiode sync square.** Draw a small black/white square in a screen
      corner that flips on key events (trial onset, each game frame). A
      photodiode taped over it gives ground-truth stimulus timing independent
      of the software clock. `mario_task` does *not* use one, but it's the gold
      standard for tying frames to the acquisition clock.
- [ ] **LSL event markers.** Push markers to a Lab Streaming Layer outlet
      (`pylsl.StreamOutlet`) at trial/phase/frame boundaries so timing lands in
      the same recording as physio/eyetracking. See `mario_task/markers.py` for
      a clean backend abstraction (LSL / serial / parallel / null with
      graceful fallback) to copy. **For fMRI the slow HRF makes the `=`-anchored
      software clock adequate; this matters most for MEG/EEG.**
- [ ] **Parallel-port / serial trigger out**, for setups that record TTL rather
      than LSL.
- [ ] Log actual vs. target frame timing (dropped-frame accounting) and
      apply drift correction against the scanner clock, as vgdl does
      (`run_time` vs. `time.time() - scan_start_ts`).

### Curriculum & design
- [ ] **Per-subject deterministic curriculum generation** (à la vgdl's
      `gen_subj` / mario's seeded design TSV): generate and persist the whole
      protocol up front, seeded by subject id, so sessions are reproducible and
      resumable.
- [ ] **Level selection.** ALE exposes game *modes* and *difficulties*
      (`ALEInterface.setMode/setDifficulty`, or Gymnasium `make(..., mode=,
      difficulty=)`); expose these per game block to build an actual difficulty
      curriculum. (Atari "levels" are usually mode/difficulty variants rather
      than discrete level files.)
- [ ] Multi-run structure with per-run trigger waits (one `=` per fMRI run),
      and inter-run rest screens.

### Presentation
- [ ] Richer instruction screens (per-game control diagrams).
- [ ] Configurable key mapping per subject (left-handed, button box layouts).
- [ ] **Button-box / MRI-safe response device** support — currently reads the
      keyboard; scanner setups often use fiber-optic button boxes that emit
      specific keycodes. Make the key set configurable.
- [ ] Audio (ALE can emit game sound; disabled here). Requires MRI-safe
      headphones and its own timing considerations.

### Data & analysis
- [ ] BIDS-style output layout (`sub-XX/ses-YY/...`) and an `events.tsv` per run
      matching BIDS task conventions, as `mario_task` produces.
- [ ] A small replay/QC utility to render a block to video and verify
      frame reconstruction against a stored checksum.
- [ ] Extract common "objects/positions" state where available (some analyses
      want sprite coordinates; ALE only gives RAM + pixels, so this would need
      per-game RAM decoding or an object-detection pass).

### Robustness
- [ ] Crash-safe incremental logging (write frames to disk as they're
      collected rather than holding a full block in memory) for long runs.
- [ ] Pause/abort screen for the experimenter mid-run.
