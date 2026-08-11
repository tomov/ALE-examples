# ALE-examples

Small standalone examples for playing and adapting
[Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
(Atari 2600) games as a **human** — including a proof-of-concept framework for
running any ALE game as an **fMRI/neuroimaging task**.

These scripts depend only on the `ale-py` PyPI package (no ALE source build
required). The Atari ROMs ship inside the `ale-py` wheel.

## Contents

| File | Purpose |
|------|---------|
| `play_atari.py`   | Play any Atari game by hand with the keyboard. |
| `fmri_play.py`    | fMRI framework: fixation → scanner trigger (`=`) → JSON curriculum → replayable per-frame logging. |
| `fMRI_README.md`  | Full docs for the fMRI framework (curriculum format, data format, engine stack, future work). |

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

See [`fMRI_README.md`](fMRI_README.md) for the fMRI framework in detail.
