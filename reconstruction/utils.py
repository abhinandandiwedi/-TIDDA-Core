from pathlib import Path


def estimate_camera_trajectory(frame_paths: list[Path], output_path: Path):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    trajectory = np.zeros((len(frame_paths), 2), dtype=float)
    previous = None
    for index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if previous is not None:
            old_points = cv2.goodFeaturesToTrack(previous, 200, 0.01, 8)
            if old_points is not None:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, image, old_points, None)
                if new_points is not None:
                    valid = status.ravel() == 1
                    delta = (new_points[valid] - old_points[valid]).mean(axis=0)
                    trajectory[index] = trajectory[index - 1] + delta
        previous = image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(trajectory[:, 0], trajectory[:, 1], color="#f97316", marker=".")
    axis.set_title("Relative camera movement")
    axis.set_xlabel("Relative X")
    axis.set_ylabel("Relative Y")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return trajectory
