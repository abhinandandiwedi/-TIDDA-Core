# Image/Video to 3D Reconstruction

## Streamlit MVP

The lightweight demo in `app.py` runs the requested in-process path:

`uploaded video -> OpenCV frame extraction -> DPT monocular depth -> Open3D PLY -> relative trajectory`

It produces a relative, local, up-to-scale reconstruction. RGB video alone cannot provide absolute geographic coordinates or metric scale. The depth model is downloaded by Transformers on first use and CUDA is selected automatically when available.

Install and run from the repository root:

```powershell
cd .\-TIDDA-Core-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r reconstruction\requirements.txt
streamlit run reconstruction\app.py
```

The app writes `pointcloud.ply` and `camera_trajectory.png` into its session workspace and exposes the PLY through the download control. The `reconstruction()` function is the integration boundary for a future Fusion Engine adapter: GPS/MAVLink pose, calibration, detections, and georeferencing can be attached to its frame/depth/point-cloud stages without changing the UI.

This package is a real orchestration layer for:

`images/video -> FFmpeg frames -> COLMAP features/matches/poses/sparse cloud -> Nerfstudio or official Gaussian Splatting -> dense model/viewer`

It does not pretend external tools are installed. Each stage validates its executable and raises a clear error. The core has no network, database, CUDA, OpenCV, or PyTorch import requirement.

## Project tree

```text
reconstruction/
  __init__.py config.py gpu.py runner.py input.py ffmpeg.py colmap.py
  nerfstudio.py gaussian.py pipeline.py main.py requirements.txt
  tests/test_pipeline.py
```

Runtime folders are created under the selected workspace: `input/`, `frames/`, `colmap/`, `sparse/`, `nerfstudio/`, and `output/`. The staged `input/` and `sparse/` layout is compatible with the official Gaussian Splatting `convert.py -s <location>` convention.

## Run

From the repository root:

```powershell
python -m reconstruction.main .\input\scene.mp4 --workspace .\reconstruction_workspace --method nerfstudio --dry-run --cpu
python -m reconstruction.main .\input\scene.mp4 --workspace .\reconstruction_workspace --method nerfstudio --train
```

For an image directory:

```powershell
python -m reconstruction.main .\input\images --workspace .\reconstruction_workspace --method nerfstudio --train
```

`--dry-run` prints the exact commands without executing tools. Remove it only after installing and validating FFmpeg, COLMAP, and Nerfstudio. CPU mode is for orchestration/testing; NeRF/Gaussian training normally needs a compatible NVIDIA GPU.

## Installation

### Windows

1. Install FFmpeg and add `ffmpeg.exe` to PATH.
2. Install COLMAP using the official Windows prebuilt binary. The official docs also document VCPKG CUDA builds:

```powershell
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install colmap[cuda,tests]:x64-windows
```

3. Use Conda for Nerfstudio. The official docs currently recommend Python >=3.8, PyTorch 2.1.2 + CUDA 11.8, CUDA toolkit, Ninja, tiny-cuda-nn, then `pip install nerfstudio`. Follow the official page because GPU wheels and versions change.
4. For original Gaussian Splatting, clone recursively and use its `environment.yml`; Windows requires Visual Studio C++ tools and the project documents CUDA 11/11.8 assumptions.

### Linux

```bash
sudo apt install ffmpeg
# COLMAP packages/build instructions: https://colmap.github.io/install.html
conda create -n nerfstudio python=3.8 -y
conda activate nerfstudio
python -m pip install --upgrade pip
# Install the CUDA-matched PyTorch command from https://pytorch.org/get-started/locally/
pip install nerfstudio
```

For the official Gaussian Splatting reference implementation:

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
```

Its documented optimizer command is `python train.py -s <COLMAP-or-NeRF-dataset>`. It lists CUDA-ready GPU, Visual Studio/g++, CUDA SDK 11.x, and high VRAM requirements for reference-quality training.

## Stages and commands

The pipeline runs these real commands:

```text
ffmpeg -i video -vf fps=2 frames/%06d.jpg
colmap feature_extractor --database_path ... --image_path ...
colmap exhaustive_matcher --database_path ...
colmap mapper --database_path ... --image_path ... --output_path ...
ns-process-data images --data ... --output-dir ...
ns-train nerfacto --data ...
python <gaussian_repo>/convert.py -s <workspace>
python <gaussian_repo>/train.py -s <workspace> -m <output>/gaussian_model
```

`check_gpu()` reports `nvidia-smi`, PyTorch installation, `torch.cuda.is_available()`, and the PyTorch CUDA runtime. CUDA-required runs fail clearly when unavailable; `--cpu` only bypasses the preflight and does not make GPU training fast or supported.

## Integration

The orchestration layer is independent from TIDDA's live WebSocket and `fusion/` packages. A later adapter can consume COLMAP/NeRF detections and send `fusion.Detection` records without changing this pipeline. Use Nerfstudio's viewer command from its installed environment for inspection; use the Gaussian Splatting SIBR viewer documented by its official repository for trained models.

## Limitations

COLMAP and Nerfstudio must be installed separately; no reliable cross-platform one-line installer exists for all CUDA/driver combinations. Gaussian Splatting's official reference implementation has older environment assumptions, so this package generates commands but does not auto-clone/install it. Metric geolocation still needs calibrated camera pose plus drone GPS/altitude; COLMAP alone is up-to-scale.
