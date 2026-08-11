"""Play an Atari game interactively as a human.

Usage:
    python play_atari.py                 # defaults to Breakout, 30 fps
    python play_atari.py ALE/Pong-v5
    python play_atari.py ALE/Pong-v5 10  # 2nd arg = fps (lower = slower/easier)

Controls:
    Arrow keys : move (UP / DOWN / LEFT / RIGHT and diagonals)
    SPACE      : fire
    Arrows + SPACE combine (e.g. RIGHT + SPACE = RIGHTFIRE)
    ESC        : quit

Atari envs have no built-in keyboard mapping, so we build `keys_to_action`
from the game's own action meanings.
"""

import sys

import ale_py
import gymnasium as gym
import pygame
from gymnasium.utils.play import play

gym.register_envs(ale_py)

env_id = sys.argv[1] if len(sys.argv) > 1 else "ALE/Breakout-v5"
# Optional 2nd arg: frames per second (lower = slower/easier). With
# frameskip=1, fps also sets game-time: the real Atari runs at 60, so 30 is
# comfortable half-speed. Going much lower stretches each game's startup
# (e.g. Pong's opponent paddle takes ~60 frames to appear).
fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

env = gym.make(
    env_id,
    render_mode="rgb_array",
    frameskip=1,                    # advance 1 frame per input (v5 default is 4)
    repeat_action_probability=0.0,  # disable "sticky actions" so keys feel responsive
)

# Map each ALE action meaning to the keys that must be held for it.
# play() matches on the sorted tuple of pressed keys, and expects pygame key
# *codes* (integers) -- not names -- so we use pygame's K_* constants.
KEY_NAMES = {
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_SPACE: "FIRE",
}
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


def keys_for_meaning(meaning):
    fire = meaning.endswith("FIRE")
    direction = meaning[:-4] if fire and meaning != "FIRE" else meaning
    keys = ()
    if direction in DIRECTION_KEYS:
        keys = DIRECTION_KEYS[direction]
    if fire:
        keys = keys + (pygame.K_SPACE,)
    return tuple(sorted(keys))


meanings = env.unwrapped.get_action_meanings()
keys_to_action = {}
for action, meaning in enumerate(meanings):
    if meaning == "NOOP":
        continue
    keys = keys_for_meaning(meaning)
    if keys:  # skip anything we couldn't map
        keys_to_action[keys] = action

print(f"Playing {env_id}.")
print("Controls:")
for keys, action in sorted(keys_to_action.items(), key=lambda kv: kv[1]):
    label = " + ".join(KEY_NAMES.get(k, str(k)) for k in keys)
    print(f"  {label:<18} -> {meanings[action]}")
print("  ESC quits.")
print(f"Speed: {fps} fps (pass a 2nd arg to change, e.g. `python play_atari.py {env_id} 10`).")

play(env, zoom=4, fps=fps, keys_to_action=keys_to_action, noop=0)
