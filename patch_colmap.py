import sys
with open('reconstruction/colmap.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if '"feature_extractor"' in line:
        lines[i] = line.replace(']', ', "--FeatureExtraction.use_gpu", "0"]')
    elif '"exhaustive_matcher"' in line or '"matcher"' in line:
        lines[i] = line.replace(']', ', "--FeatureMatching.use_gpu", "0"]')
with open('reconstruction/colmap.py', 'w') as f:
    f.writelines(lines)
