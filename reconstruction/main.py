import argparse
import logging
from pathlib import Path

from .config import ReconstructionConfig
from .gpu import check_gpu
from .pipeline import ReconstructionPipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Video/images -> COLMAP -> NeRF or Gaussian Splatting")
    result.add_argument("input", type=Path, help="image folder or video file")
    result.add_argument("--workspace", type=Path, default=Path("reconstruction_workspace"))
    result.add_argument("--method", choices=("nerfstudio", "gaussian"), default="nerfstudio")
    result.add_argument("--fps", type=float, default=2.0)
    result.add_argument("--train", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--cpu", action="store_true", help="skip the CUDA requirement")
    result.add_argument("--colmap", default="colmap")
    result.add_argument("--ffmpeg", default="ffmpeg")
    result.add_argument("--gaussian-repo", type=Path)
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args()
    print(check_gpu())
    config = ReconstructionConfig(args.workspace, args.input, args.fps, args.colmap, args.ffmpeg,
                                  gaussian_repo=args.gaussian_repo, use_cuda=not args.cpu)
    try:
        result = ReconstructionPipeline(config, dry_run=args.dry_run).run(args.method, args.train)
    except (OSError, RuntimeError, ValueError) as exc:
        logging.error("pipeline failed: %s", exc)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
