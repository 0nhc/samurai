# GUI Demo for SAMURAI
 This is my modified version of SAMURAI with a GUI to tracker multiple objects.
- **Purpose:** : A small interactive demo that lets you select one-or-multi object bounding boxes on the first frame and runs SAMURAI's video-mask propagation to produce a video with a pure green background and the selected objects composited back in.
- **Location:** : `gui_demo.py` at the repository root.
- **Key features:** : Click-and-drag bounding-box selection; supports input as an `.mp4` video or a directory of image frames; multi-object mask propagation via SAM 2.1-based predictor; outputs a composited video with a green background.

## Prerequisites
I only tested with:
* `Ubuntu 22.04`
* `Pyhton 3.10`
* `CUDA 12.1`

If it doesn't work on your system, consider using `conda` or `mamba` to create a virtual environment with `Python 3.10` and `CUDA 12.1`.
## Installation
```sh
# Clone this repo
git clone https://github.com/0nhc/samurai.git

# Create a virtual Python environment. You can also use conda or mamba.
cd samurai
python3 -m venv samurai_venv
source samurai_venv/bin/activate

# Install SAM2
cd sam2
pip install -e .
pip install -e ".[notebooks]"
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru typeguard pyyaml
cd checkpoints
./download_ckpts.sh # Download SAM2 Checkpoints
cd ../..
```

## Quick Start
```sh
# Make sure the virtual environment is sourced
source samurai_venv/bin/activate
python gui_demo.py --video_path path/to/video.mp4
```
Detailed arguments:
- `--video_path` : Path to input `.mp4`
- `--model_path` : Path to SAM 2.1 checkpoint (e.g. `sam2/checkpoints/sam2.1_hiera_base_plus.pt`). The script auto-selects a config by name (large/base_plus/small/tiny).
- `--output_path` : Path to write the output video (default `output_green_background.mp4`).

## Original Repository's README
See [README](ORIGINAL_README.md).