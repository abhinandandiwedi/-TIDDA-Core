#!/bin/bash
set -e
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

echo "=== TIDDA REAL 3D MAPPING DEMO V3 (DENSE) ==="

WORKSPACE="workspace_mapping_demo_v3"
COLMAP_BIN="./colmap_hip_build/build/src/colmap/exe/colmap"
VIDEO="/home/abhinandan/Downloads/videoplayback.mp4"

# Resuming Dense Reconstruction
echo "Running PatchMatch Stereo..."
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
${COLMAP_BIN} patch_match_stereo \
    --workspace_path ${WORKSPACE}/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0

echo "Running Stereo Fusion..."
${COLMAP_BIN} stereo_fusion \
    --workspace_path ${WORKSPACE}/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path ${WORKSPACE}/output/final_map.ply

echo "=== PIPELINE COMPLETE ==="
