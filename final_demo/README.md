# Costume Recognition — Live GUI Demo

Live cosplay/costume recognition on a Jetson Orin (ZED Box Orin NX), built
for DiMi (TIAGo robot) at Fantasy Basel. A ZED X camera feeds a live PyQt6
GUI that detects people in real time and, on demand, classifies each
detected costume against predefined character classes.

## Files

| File | Role |
|---|---|
| `gui.py` | PyQt6 app — live feed, Capture button, results panel |
| `camera_session.py` | ZED camera + person detector wrapper (`CameraSession`) |
| `clip_classifier.py` | SigLIP2 zero-shot/finetuned classifier (`ClipCostumeClassifier`) |
| `labels.json` | The 30 character class names (source of truth for labels) |
| `40c_epoch8.pt` | Finetuned SigLIP2 checkpoint — **must be in this folder** (see below) |
| `requirements.txt` | Jetson-specific install instructions |

## Setup

Target hardware: ZED Box Orin NX 16GB, JetPack 6.1, L4T 36.4, CUDA 12.6, Python 3.10.

```bash
pip install -r requirements.txt
```

This pulls Jetson-native `torch`/`torchvision` wheels from the Jetson AI
Lab index, plus OpenCLIP, PyQt6, and imaging dependencies. `pyzed` (the ZED
SDK Python API) is **not** pip-installable — install the ZED SDK first,
then run:

```bash
python /usr/local/zed/get_python_api.py
```

Verify both installs:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # -> True
python -c "import pyzed.sl as sl; print(sl.Camera.get_sdk_version())"
```

If PyQt6 won't install via pip on aarch64, use the system package instead
(`sudo apt-get install -y python3-pyqt6`) and create your venv with
`--system-site-packages`.

### Checkpoint

Place `40c_epoch8.pt` in the same folder as `gui.py`/`clip_classifier.py`.
The checkpoint filename is hardcoded as `CHECKPOINT` in both `gui.py` and
`clip_classifier.py` — update it there if you use a different checkpoint.

## Running the live GUI

```bash
python gui.py
```

This opens the ZED camera, loads the SigLIP2 model (takes a few seconds),
then launches the app maximized. Live green boxes track detected people;
click **Capture** to classify everyone currently in frame.

**Display options** (right panel, no re-classification needed to change):
- **Percentage / Cosine similarity** toggle — switch between softmax
  confidence and raw cosine similarity for the same predictions.
- **Costume threshold slider** — sets the top-1 softmax % below which a
  person is shown as "Not recognized" rather than assigned a guess.

## Running the classifier standalone (no camera)

Useful for testing the model against a folder of pre-saved crops:

```bash
python clip_classifier.py --crops-dir crops [--topk 3] [--cpu]
```