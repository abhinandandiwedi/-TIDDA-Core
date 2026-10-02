<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=TIDDA&fontSize=72&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Tactical%20Intelligent%20Drone%20Defense%20Architecture&descAlignY=60&descSize=18" alt="TIDDA banner" width="100%"/>

<a href="https://github.com/abhinandandiwedi/-TIDDA-Core">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&pause=1200&color=36BCF7&center=true&vCenter=true&width=640&lines=Perceive.+Map.+Remember.+Localize.+Coordinate.;Video+%E2%86%92+3D+reconstruction+%E2%86%92+persistent+world;SIFT+%7C+ALIKED+%2B+LightGlue+%7C+COLMAP+%7C+ROCm%2FHIP" alt="Typing tagline"/>
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ROCm](https://img.shields.io/badge/AMD-ROCm%20%2F%20HIP-ED1C24?style=for-the-badge&logo=amd&logoColor=white)
![Phase](https://img.shields.io/badge/Phase%207-Complete-success?style=for-the-badge)
![Next](https://img.shields.io/badge/Next-Phase%208-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active%20Research-yellow?style=for-the-badge)

**A software-first research platform for human-supervised command & control, distributed visual nodes, 3D reconstruction, cooperative mapping, and persistent spatial memory.**

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Status Legend](#-status-legend)
3. [At a Glance](#-at-a-glance)
4. [System Architecture](#-system-architecture)
5. [Spatial Intelligence Pipeline](#-spatial-intelligence-pipeline)
6. [Command & Control](#-command--control)
7. [Mobile / Edge Nodes](#-mobile--edge-nodes)
8. [Video → 3D Reconstruction](#-video--3d-reconstruction)
9. [Advanced Perception](#-advanced-perception-sift--aliked--lightglue)
10. [Incremental 3D Mapping](#-incremental-3d-mapping)
11. [Cooperative Multi-Node Mapping](#-cooperative-multi-node-mapping)
12. [Persistent 3D World](#-persistent-3d-world)
13. [2D → 3D Object Localization](#-2d--3d-object-localization)
14. [Real-World Validation](#-real-world-validation)
15. [Testing Methodology](#-testing-methodology)
16. [Roadmap](#-roadmap)
17. [Project Structure](#-project-structure)
18. [Technology Stack](#-technology-stack)
19. [Getting Started](#-getting-started)
20. [Screenshots](#-screenshots)
21. [Engineering Goals & Design Principles](#-engineering-goals--design-principles)
22. [Future Integrations](#-future-integrations)
23. [Author](#-author)
24. [Safety & Research Disclaimer](#-safety--research-disclaimer)
25. [Vision](#-vision)

---

## 🧭 Overview

TIDDA started as a drone-swarm command-and-control simulator and has grown into a computer-vision and spatial-intelligence stack. Its core is a pipeline that turns video from distributed camera nodes into 3D structure, keeps that structure as persistent state, and exposes it to a human operator through a live dashboard.

TIDDA combines several distinct disciplines. They are kept separate on purpose:

| Discipline | Used for |
|---|---|
| Classical computer vision | SIFT and ORB features, Essential Matrix, RANSAC, triangulation, ICP |
| Learned features | ALIKED keypoints/descriptors, LightGlue matching |
| Photogrammetry / SfM | COLMAP sparse and dense reconstruction |
| Learned detection | YOLO 2D object detection |
| Distributed systems | WebSocket C2 backend, node registry, heartbeats |

**What TIDDA is not:** it is not a flight controller, not a production SLAM system, and not a deployed military system. See [Safety & Research Disclaimer](#-safety--research-disclaimer).

---

## 🏷️ Status Legend

| Tag | Meaning |
|---|---|
| **IMPLEMENTED** | Code exists in the repository |
| **VALIDATED** | Run on real data with recorded results |
| **EXPERIMENTAL** | Works in limited conditions, not hardened |
| **PLANNED** | Not implemented; roadmap only |

---

## 📊 At a Glance

### Development progress (by phase)

```mermaid
pie showData title Roadmap phases by status
    "Complete (1 to 7, incl. 4.5 / 5.5 / 6.5)" : 10
    "Next (Phase 8)" : 1
    "Planned (Phases 9 and 10)" : 2
```

<!-- VERIFY: counts assume phases 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7 are complete -->

### Key capabilities

| Capability | Status |
|---|---|
| WebSocket C2 backend with node registry and heartbeat monitoring | IMPLEMENTED |
| Browser-based phone node client (`html/phone_client.html`) | IMPLEMENTED |
| Video ingest and frame sampling (`ingest/`) | IMPLEMENTED |
| COLMAP sparse + dense reconstruction (HIP build supported) | IMPLEMENTED, VALIDATED |
| Selectable feature pipeline: `sift`, `aliked`, `aliked_lightglue` | IMPLEMENTED, VALIDATED |
| YOLO 2D detection and 2D → 3D localization | IMPLEMENTED <!-- VERIFY: Phase 3 --> |
| Incremental 3D mapping (ORB, Essential Matrix, RANSAC, triangulation) | IMPLEMENTED <!-- VERIFY: Phase 4/4.5 --> |
| Cooperative multi-node mapping (alignment + ICP + global fusion) | IMPLEMENTED <!-- VERIFY: Phase 5/5.5 --> |
| Persistent 3D world (storage, versioning, save/reload, recovery) | IMPLEMENTED <!-- VERIFY: Phase 6/6.5 --> |
| Persistent localization & relocalization (PnP + RANSAC) | **PLANNED** (Phase 8) |
| Real drone hardware, PX4/MAVLink, mesh networking, mission planning | **PLANNED / not implemented** |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    OP([👤 Human Operator]) --> DASH[🖥️ TIDDA Tactical Dashboard<br/>HTML / JavaScript]
    DASH <-->|WebSocket| BE[⚙️ Python Backend<br/>FastAPI / WebSockets]
    BE <--> NODES[📱 Distributed Nodes<br/>browser phone clients, simulated nodes]
    NODES --> PERC[👁️ Perception<br/>features, matching, YOLO]
    PERC --> MAP[🗺️ Mapping<br/>incremental + cooperative]
    MAP --> LOC[📍 2D → 3D Localization]
    MAP --> WORLD[(💾 Persistent 3D World)]
    LOC --> WORLD
    WORLD --> DASH

    classDef op fill:#1f6feb,stroke:#fff,color:#fff
    classDef core fill:#203a43,stroke:#36bcf7,color:#fff
    classDef store fill:#2d6a4f,stroke:#95d5b2,color:#fff
    class OP op
    class DASH,BE,NODES,PERC,MAP,LOC core
    class WORLD store
```

---

## 🔁 Spatial Intelligence Pipeline

```mermaid
flowchart LR
    A[👁️ Perceive] --> B[🗺️ Map] --> C[💾 Remember] --> D[📍 Localize] --> E[🤝 Coordinate]
    E -. feedback .-> A
    style A fill:#264653,color:#fff
    style B fill:#2a9d8f,color:#fff
    style C fill:#e9c46a,color:#000
    style D fill:#f4a261,color:#000
    style E fill:#e76f51,color:#fff
```

| Stage | Meaning in TIDDA |
|---|---|
| Perceive | Feature extraction, matching, 2D detection |
| Map | Sparse/dense reconstruction, incremental mapping |
| Remember | Persistent world store, map versioning |
| Localize | 2D → 3D object localization today; camera relocalization planned (Phase 8) |
| Coordinate | Operator-facing C2 and multi-node fusion |

---

## 🎛️ Command & Control

- Python backend with WebSocket communication (`tidda_lightweight_swarm.py`, `core/`)
- Node registration, heartbeat monitoring and disconnect detection (`mobile_node.py`)
- Typed message schema in `core/message_schema.py` <!-- VERIFY -->
- Live HTML/JavaScript dashboard (`dashboard/index.html`) <!-- VERIFY: Three.js 3D viewer -->
- Reconstructed point clouds served to the dashboard from `/models/fused.ply`

The repository also contains a swarm simulator (`simulator.py`, `swarm_logic.py`) that predates the vision stack.

```mermaid
sequenceDiagram
    participant P as 📱 Phone Node
    participant B as ⚙️ Backend
    participant D as 🖥️ Dashboard
    P->>B: register (WebSocket)
    B-->>D: node online
    loop heartbeat
        P->>B: heartbeat + telemetry
        B-->>D: live update
    end
    P->>B: upload video
    B->>B: reconstruction job
    B-->>D: fused.ply ready
    Note over B,P: missed heartbeats → disconnect detected
```

---

## 📱 Mobile / Edge Nodes

Phones join as nodes through a browser client (`html/phone_client.html`, served over HTTPS by `serve_phone_https.py`). A node can register, send heartbeats and upload video for reconstruction.

Real drone hardware is **not** integrated.

---

## 🎥 Video → 3D Reconstruction

`run_pipeline.py` runs the offline reconstruction pipeline:

```mermaid
flowchart LR
    V[🎞️ Video] --> F[Frame sampling<br/>ingest/] --> X[Feature extraction<br/>+ matching]
    X --> G[Geometric<br/>verification] --> S[COLMAP<br/>sparse] --> DN[COLMAP<br/>dense] --> PLY[(fused.ply)] --> DASH[🖥️ Dashboard]
```

```bash
python run_pipeline.py <video.mp4> --method=sift
python run_pipeline.py <video.mp4> --method=aliked
python run_pipeline.py <video.mp4> --method=aliked_lightglue
```

The feature method can also be set with the `TIDDA_FEATURE_METHOD` environment variable (default: `sift`). The COLMAP binary is taken from `COLMAP_EXEC`, defaulting to the local HIP build under `colmap_hip_build/`.

---

## 🧠 Advanced Perception: SIFT / ALIKED / LightGlue

| Method | Type | Role in TIDDA |
|---|---|---|
| SIFT | Classical hand-crafted features | Baseline |
| ALIKED | Learned keypoints + descriptors | Alternative extractor |
| LightGlue | Learned matcher | Matches ALIKED features |
| Geometric verification | Epipolar / RANSAC filtering | Rejects false matches before SfM |

```mermaid
flowchart LR
    CAM[📷 Camera] --> EXT{{SIFT / ALIKED}} --> MATCH[LightGlue /<br/>classical matching] --> GV[Geometric<br/>Verification] --> COL[COLMAP] --> R3D[(3D Reconstruction)]
```

GPU execution (PyTorch ROCm for ALIKED/LightGlue, HIP build of COLMAP for dense reconstruction) has been run on an AMD Radeon RX 9060 XT (gfx1200). CPU fallback is supported <!-- VERIFY -->.

---

## 🧩 Incremental 3D Mapping

Real-time map growth from a moving camera:

```mermaid
flowchart LR
    F1[Frame t] --> ORB[ORB tracking] --> E[Essential Matrix<br/>+ RANSAC] --> POSE[Pose recovery] --> TRI[Triangulation] --> FUSE[Temporal fusion] --> MAP[(Running map)]
    E -. low quality .-> FAIL[Tracking-failure<br/>handling]
    FAIL -.-> ORB
```

This is visual incremental mapping. It is **not** a full SLAM system: there is no loop closure or global bundle adjustment in the real-time path <!-- VERIFY -->.

---

## 🤝 Cooperative Multi-Node Mapping

```mermaid
flowchart TD
    A[Node A] --> MA[Local Map A]
    B[Node B] --> MB[Local Map B]
    MA --> XM[Cross-node matching]
    MB --> XM
    XM --> AL[Alignment]
    AL --> ICP[Multi-scale ICP]
    ICP --> QC{Alignment<br/>quality OK?}
    QC -- yes --> GM[(Global Map)]
    QC -- no --> REJ[Reject / retry]
```

<!-- VERIFY: Phase 5.5 multi-scale ICP and quality checks -->

---

## 💾 Persistent 3D World

- World state stored on disk (`workspaces/default/world`, `world_model.py`, `reconstruction/world_store.py` <!-- VERIFY -->)
- Map versioning, save and reload
- Validation of corrupt or missing state, with recovery
- Node re-registration after restart

---

## 🎯 2D → 3D Object Localization

YOLO gives a 2D detection `(u, v)`. With depth `Z` and camera intrinsics `fx, fy, cx, cy`, the camera-frame point is:

```
Xc = (u − cx) · Z / fx
Yc = (v − cy) · Z / fy
Zc = Z
```

COLMAP poses follow `Xc = R·Xw + t`, so the world-frame point is:

```
Xw = Rᵀ (Xc − t)
```

<!-- VERIFY: confirm against reconstruction/localization_3d.py -->

---

## 🌍 Real-World Validation

Results below are measured on real video. They are specific to these datasets and should not be read as general accuracy claims.

### House V1 (indoor)

Real indoor video run through SIFT baseline, ALIKED, LightGlue, COLMAP dense reconstruction, 2D → 3D localization and incremental mapping. Logs and workspaces are in `workspace_house_v1_*` and `pipeline_house_v1.log`.

### Iceland FPV drone footage

1920×1080, 30 FPS, ~67 s, 336 sampled frames. <!-- VERIFY all numbers against logs -->

| Metric | SIFT | ALIKED + LightGlue |
|---|---|---|
| Registered cameras | 335 / 336 (99.70%) | 335 / 336 (99.70%) |
| Keypoints | 1,670,400 | 680,485 |
| Matched pairs | 42,840 | 55,374 |
| Geometrically verified pairs | 7,123 | 19,046 |
| Geometric inliers | 1,024,607 | 2,809,123 |
| Sparse 3D points | 66,912 | 100,617 |

Additional ALIKED + LightGlue statistics: 1 connected component, 493,244 observations/tracks, mean track length 4.90, mean reprojection error 1.2738 px.

#### Camera registration (both methods)

```mermaid
pie showData title Cameras registered out of 336 sampled frames
    "Registered" : 335
    "Not registered" : 1
```

#### SIFT vs ALIKED + LightGlue

```mermaid
xychart-beta
    title "Keypoints extracted"
    x-axis ["SIFT", "ALIKED"]
    y-axis "keypoints" 0 --> 1800000
    bar [1670400, 680485]
```

```mermaid
xychart-beta
    title "Matched pairs"
    x-axis ["SIFT", "ALIKED + LightGlue"]
    y-axis "pairs" 0 --> 60000
    bar [42840, 55374]
```

```mermaid
xychart-beta
    title "Geometrically verified pairs"
    x-axis ["SIFT", "ALIKED + LightGlue"]
    y-axis "pairs" 0 --> 20000
    bar [7123, 19046]
```

```mermaid
xychart-beta
    title "Sparse 3D points"
    x-axis ["SIFT", "ALIKED + LightGlue"]
    y-axis "points" 0 --> 110000
    bar [66912, 100617]
```

**Reading the comparison:** ALIKED used fewer keypoints but produced more verified pairs and a denser sparse model on this sequence. Both registered the same number of cameras. No runtime or memory comparison is reported here, and results on other scenes may differ. This is one dataset, not evidence that one method is universally better.

---

## 🧪 Testing Methodology

- `pytest` suites in `tests/` and root-level `test_*.py` files (fusion, local map, mapping pipeline, mobile node, perception, scan session, world model, floor manager, GPU check)
- Real-video validation runs with logged metrics
- Persistence regression tests covering corrupt/missing state <!-- VERIFY -->

```bash
pytest
```

---

## 🗺️ Roadmap

```mermaid
flowchart LR
    P1[1 C2 Foundation]:::done --> P2[2 Mobile Nodes]:::done --> P3[3 2D→3D Localization]:::done --> P4[4 Incremental Mapping]:::done --> P45[4.5 Tracking Robustness]:::done
    P45 --> P5[5 Cooperative Mapping]:::done --> P55[5.5 Mapping Optimization]:::done --> P6[6 Persistent World]:::done --> P65[6.5 Persistence Hardening]:::done --> P7[7 Advanced Perception]:::done
    P7 --> P8[8 Persistent Localization]:::next --> P9[9 Dynamic World / Multi-Agent]:::planned --> P10[10 End-to-End Validation]:::planned

    classDef done fill:#2d6a4f,stroke:#95d5b2,color:#fff
    classDef next fill:#1f6feb,stroke:#fff,color:#fff,stroke-width:3px
    classDef planned fill:#495057,stroke:#adb5bd,color:#fff,stroke-dasharray: 4 3
```

🟩 Complete  🟦 Next  ⬜ Planned

| Phase | Name | Status |
|---|---|---|
| 1 | C2 Foundation (backend, WebSocket, reconstruction integration) | ✅ Complete |
| 2 | Mobile Nodes (phone video upload, reconstruction job API) | ✅ Complete |
| 3 | 2D → 3D Localization (YOLO, depth, COLMAP pose) | ✅ Complete |
| 4 | Incremental 3D Mapping | ✅ Complete |
| 4.5 | Tracking Robustness | ✅ Complete |
| 5 | Cooperative Mapping | ✅ Complete |
| 5.5 | Mapping Optimization | ✅ Complete |
| 6 | Persistent World | ✅ Complete |
| 6.5 | Persistence Hardening | ✅ Complete |
| 7 | Advanced Perception (SIFT, ALIKED, LightGlue, ROCm/HIP) | ✅ **Complete** |
| 8 | Persistent Localization & Relocalization | 🔵 **Next (planned, not implemented)** |
| 9 | Dynamic World / Multi-Agent Intelligence | ⬜ Planned |
| 10 | End-to-End Validation | ⬜ Planned |

<!-- VERIFY each "Complete" against git history before publishing -->

<details>
<summary><b>Phase 8 — planned architecture (click to expand)</b></summary>

```mermaid
flowchart TD
    A[New live camera frame] --> B[ALIKED feature extraction]
    B --> C[LightGlue matching]
    C --> D[(Known persistent 3D world)]
    D --> E[2D ↔ 3D correspondences]
    E --> F[PnP + RANSAC]
    F --> G[Camera pose]
    G --> H[Persistent localization]
    H --> I[Relocalization after tracking loss]
    classDef plan fill:#495057,stroke:#adb5bd,color:#fff,stroke-dasharray: 4 3
    class A,B,C,D,E,F,G,H,I plan
```

Nothing in this diagram is implemented yet.

</details>

---

## 🗂️ Project Structure

<details open>
<summary><b>Repository layout</b></summary>

```
TIDDA-Core/
├── core/                      # Backend core (message schema, server)
├── dashboard/                 # Operator dashboard (HTML/JS)
├── html/                      # Browser clients (phone_client.html)
├── ingest/                    # Video ingest, frame sampling
├── reconstruction/            # SfM / mapping / localization modules
│   ├── colmap_runner.py       # COLMAP sparse pipeline (feature_method selectable)
│   ├── dense_reconstruction.py
│   ├── features.py            # <!-- VERIFY -->
│   ├── matching.py            # <!-- VERIFY -->
│   ├── incremental_mapper.py  # <!-- VERIFY -->
│   ├── cooperative_mapper.py  # <!-- VERIFY -->
│   ├── localization_3d.py     # <!-- VERIFY -->
│   ├── world_store.py         # <!-- VERIFY -->
│   └── orchestrator.py        # <!-- VERIFY -->
├── vision/                    # Vision modules
├── detection_fusion/          # Detection fusion
├── models_served/             # Served model/point-cloud output
├── workspaces/default/world/  # Persistent world state
├── tests/                     # Test suite
├── docs/                      # Documentation
├── legacy/                    # Superseded code
├── test_dataset/              # Sample data
├── colmap_hip_build/          # COLMAP HIP build tree
├── compile_colmap_hip.sh      # Build script for COLMAP with HIP
├── run_pipeline.py            # Video → sparse → dense pipeline
├── tidda_lightweight_swarm.py # Backend entry point
├── mobile_node.py             # Mobile node registry / heartbeat
├── serve_phone_https.py       # HTTPS server for phone client
├── world_model.py, local_map.py, mapping_pipeline.py, fusion*.py
├── simulator.py, swarm_logic.py
├── weapon_systems.py          # Simulation-only module (see disclaimer)
└── requirements.txt
```

</details>

---

## 🛠️ Technology Stack

| Area | Technologies |
|---|---|
| Backend | Python, FastAPI, uvicorn, websockets |
| Computer vision | OpenCV, COLMAP, SIFT, ORB, ALIKED, LightGlue, Open3D |
| Detection | YOLO (Ultralytics, `yolov8n.pt`) |
| GPU | AMD ROCm / HIP, PyTorch |
| Frontend | HTML, JavaScript, WebSocket <!-- VERIFY: Three.js --> |
| Storage | Files on disk (PLY point clouds, world state) <!-- VERIFY: JSON / SQLite --> |
| Testing | pytest |

---

## 🚀 Getting Started

```bash
git clone https://github.com/abhinandandiwedi/-TIDDA-Core.git
cd -TIDDA-Core
pip install -r requirements.txt
```

ALIKED, LightGlue and the HIP COLMAP build need additional setup (PyTorch ROCm, `compile_colmap_hip.sh`). Backend launchers: `run_tidda.sh` / `run_tidda.bat`.

---

## 🖼️ Screenshots

> Placeholders. These image files do not exist yet.

| View | Path |
|---|---|
| Tactical dashboard | `docs/images/dashboard.png` |
| 3D reconstruction | `docs/images/3d_reconstruction.png` |
| Cooperative mapping | `docs/images/cooperative_mapping.png` |
| Mobile node | `docs/images/mobile_node.png` |

---

## 🎯 Engineering Goals & Design Principles

- **Accuracy over hype:** claims are tied to code or logged results
- **Software-first:** subsystems are built and validated on real recorded data before hardware
- **Measurable:** registration rate, verified pairs, reprojection error
- **Modular:** perception, mapping, persistence and C2 are separable
- **Human-supervised:** the operator stays in the loop
- **Honest failure handling:** tracking loss and corrupt state are handled explicitly

---

## 🔭 Future Integrations

All of the following are **planned and not implemented**: PX4 / MAVLink, real drone hardware, mesh networking, sensor fusion beyond vision, autonomous mission planning, persistent relocalization (Phase 8).

---

## 👤 Author

**Abhinandan Diwedi** — Mechanical Engineering student building robotics and computer-vision software.

**Project status:** active research and development. Phases 1–7 are reported complete; Phase 8 (persistent localization & relocalization) is next.

---

## ⚠️ Safety & Research Disclaimer

TIDDA is a research and software engineering project focused on distributed robotics software, computer vision, 3D reconstruction, spatial intelligence and human-supervised command and control. It is not a deployed or operational system, has no hardware effector or weapon-control interface, and is not intended for operational use.

The repository contains `weapon_systems.py`, a simulation-only software module that models an engagement workflow state machine. It is unrelated to the perception, mapping and localization pipeline, is not part of the validated work described above, and does not connect to any hardware.

---

## 🌌 Vision

A spatial-intelligence stack where distributed cameras build, share and remember a consistent 3D picture of the world, so a human operator can understand and coordinate what is happening within it.

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,100:0f2027&height=100&section=footer" alt="footer" width="100%"/>

</div>
