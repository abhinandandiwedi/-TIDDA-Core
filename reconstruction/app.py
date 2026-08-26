import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root = str(PROJECT_ROOT)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

import streamlit as st

from reconstruction.reconstruction import reconstruct
from reconstruction.video_processor import get_video_info


st.set_page_config(page_title="Drone Video -> 3D Mapping Engine", layout="wide")
st.title("Drone Video -> 3D Mapping Engine")
st.caption("Relative/local reconstruction from monocular video. GPS, absolute scale, and georeferencing are not inferred.")

uploaded = st.file_uploader("Upload Drone Video", type=["mp4", "avi", "mov", "mkv"])
interval = st.slider("Extract every Nth frame", 1, 30, 5)
voxel_size = st.slider("Point-cloud voxel size", 0.01, 0.15, 0.03, 0.01)

if uploaded:
    root = Path(tempfile.mkdtemp(prefix="drone3d_"))
    video_path = root / uploaded.name
    video_path.write_bytes(uploaded.getbuffer())
    try:
        info = get_video_info(video_path)
        st.subheader("Video Information")
        metrics = st.columns(5)
        metrics[0].metric("FPS", f"{info.fps:.2f}")
        metrics[1].metric("Total frames", info.total_frames)
        metrics[2].metric("Resolution", f"{info.width} x {info.height}")
        metrics[3].metric("Duration", f"{info.duration_seconds:.1f}s")
        if st.button("Build 3D reconstruction", type="primary"):
            progress_bar = st.progress(0, text="Loading depth model...")
            result = reconstruct(video_path, root, interval, voxel_size=voxel_size,
                                 progress=lambda done, total: progress_bar.progress(done / total, text=f"Depth estimation: {done}/{total}"))
            st.session_state["result"] = result
            st.success("Reconstruction complete")
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")

result = st.session_state.get("result")
if result:
    st.subheader("Results")
    columns = st.columns(4)
    columns[0].metric("Frames processed", len(result.frame_paths))
    columns[1].metric("Point count", result.point_count)
    columns[2].metric("Processing time", f"{result.elapsed_seconds:.1f}s")
    columns[3].metric("Device", result.device)
    import open3d as o3d
    import plotly.graph_objects as go

    cloud = o3d.io.read_point_cloud(str(result.pointcloud_path))
    points = cloud.points
    colors = cloud.colors
    figure = go.Figure(go.Scatter3d(
        x=[point[0] for point in points], y=[point[1] for point in points], z=[point[2] for point in points],
        mode="markers", marker={"size": 2, "color": [color for color in colors], "opacity": 0.8},
    ))
    figure.update_layout(height=600, margin={"l": 0, "r": 0, "t": 20, "b": 0},
                         scene={"aspectmode": "data"})
    st.subheader("Point Cloud Preview")
    st.plotly_chart(figure, use_container_width=True)
    st.image(str(result.trajectory_path), caption="Relative camera trajectory")
    with open(result.pointcloud_path, "rb") as pointcloud_file:
        st.download_button("Download pointcloud.ply", pointcloud_file, file_name="pointcloud.ply")
    st.info("The PLY is generated from uploaded frames and estimated depth. It is local and up-to-scale.")
