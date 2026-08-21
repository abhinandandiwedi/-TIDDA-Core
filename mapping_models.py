# ══════════════════════════════════════════════════════════════════
#  🗺 TIDDA MAPPING MODELS — Data structures for cooperative mapping
#  Pure data models only. No AI, no SLAM, no fusion logic.
#  These structures will be populated by future perception and
#  mapping subsystems.
#
#  Conventions:
#    - All spatial coordinates (x, y, z) are LOCAL coordinates
#      unless explicitly annotated as global/GPS.
#    - GPS fields (latitude, longitude, altitude) are always Optional
#      because indoor environments may have no GPS fix.
#    - Timestamps are Unix epoch floats (time.time()).
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════════

class FloorStatus(enum.Enum):
    """Mapping status for a single floor."""
    NOT_STARTED = "NOT_STARTED"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"
    PAUSED = "PAUSED"


# ══════════════════════════════════════════════════════════════════
#  1. POSE — 6-DOF position + orientation + optional GPS
# ══════════════════════════════════════════════════════════════════

@dataclass
class Pose:
    """Six-degree-of-freedom pose in LOCAL coordinates.

    Spatial fields (x, y, z) are in meters relative to a local
    coordinate frame origin.  Orientation (yaw, pitch, roll) is
    in radians.

    GPS fields are Optional — they will be None when operating
    indoors or when the device has no satellite fix.
    """

    # ── Local position (meters) ──────────────────────────────────
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # ── Orientation (radians) ────────────────────────────────────
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # ── Timestamp (Unix epoch) ───────────────────────────────────
    timestamp: float = field(default_factory=time.time)

    # ── Optional global GPS location ─────────────────────────────
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None


# ══════════════════════════════════════════════════════════════════
#  2. SEMANTIC OBSERVATION — Something perceived by a phone camera
# ══════════════════════════════════════════════════════════════════

@dataclass
class SemanticObservation:
    """A single semantic observation from a phone's camera.

    Represents a classified element detected in a camera frame.
    This is the DATA STRUCTURE only — actual AI perception is
    not implemented here.

    Supported class_name values (non-exhaustive):
        wall, doorway, corridor, room, staircase,
        elevator, obstacle, object
    """

    # ── Required fields ──────────────────────────────────────────
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""
    timestamp: float = field(default_factory=time.time)
    floor_id: str = ""
    class_name: str = ""        # e.g. "wall", "doorway", "corridor"
    confidence: float = 0.0     # [0.0, 1.0]

    # ── Optional fields ──────────────────────────────────────────
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    pose: Optional[Pose] = None                    # observer pose when captured
    position: Optional[Tuple[float, float, float]] = None  # estimated (x, y, z) of observed object
    spatial_extent: Optional[Tuple[float, float, float]] = None  # (width, height, depth) meters
    frame_id: Optional[str] = None                 # camera frame identifier


# ══════════════════════════════════════════════════════════════════
#  3. LANDMARK — A persistent spatial feature
# ══════════════════════════════════════════════════════════════════

@dataclass
class Landmark:
    """A persistent, identified feature in the environment.

    Landmarks are derived from observations and serve as anchor
    points for map alignment.  No feature extraction or computer
    vision is implemented here — this is the data container only.
    """

    landmark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""               # source node that first observed this
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) local coords
    confidence: float = 0.0         # [0.0, 1.0]
    observation_count: int = 0      # how many times this landmark has been observed
    descriptor: Optional[str] = None          # optional feature descriptor reference
    reference_frame_id: Optional[str] = None  # optional reference frame


# ══════════════════════════════════════════════════════════════════
#  4. LOCAL MAP STATE — Map owned by one phone node
# ══════════════════════════════════════════════════════════════════

@dataclass
class LocalMapState:
    """The local map maintained by a single phone node.

    Preserves source node ownership — every local map belongs to
    exactly one node_id.  The trajectory is an ordered sequence of
    Pose snapshots recording the node's path.
    """

    node_id: str = ""
    floor_id: str = ""
    trajectory: List[Pose] = field(default_factory=list)
    landmarks: List[Landmark] = field(default_factory=list)
    observations: List[SemanticObservation] = field(default_factory=list)
    mapping_confidence: float = 0.0  # overall confidence [0.0, 1.0]
    coverage: float = 0.0           # fraction of assigned area covered [0.0, 1.0]
    last_update: float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════
#  5. FLOOR STATE — Aggregated state for one floor
# ══════════════════════════════════════════════════════════════════

@dataclass
class FloorState:
    """Aggregated mapping state for a single floor.

    IMPORTANT: Multiple floors can be ACTIVE simultaneously.
    This is NOT a singleton — each floor has its own independent
    FloorState instance with its own status.
    """

    floor_id: str = ""
    status: FloorStatus = FloorStatus.NOT_STARTED
    node_ids: List[str] = field(default_factory=list)
    local_maps: List[LocalMapState] = field(default_factory=list)
    landmarks: List[Landmark] = field(default_factory=list)
    observations: List[SemanticObservation] = field(default_factory=list)
    coverage: float = 0.0       # overall floor coverage [0.0, 1.0]
    confidence: float = 0.0     # overall floor confidence [0.0, 1.0]
    observation_count: int = 0  # total observations on this floor
    landmark_count: int = 0     # total landmarks on this floor
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════
#  6. WORLD MODEL STATE — Complete building model
# ══════════════════════════════════════════════════════════════════

@dataclass
class WorldModelState:
    """The complete world model representing an entire building.

    Contains N floors (Floor 1 … Floor N), all of which can
    coexist simultaneously.  Preserves source attribution via
    per-floor node_ids and per-local-map node ownership.
    """

    building_id: str = ""
    floors: Dict[str, FloorState] = field(default_factory=dict)  # floor_id → FloorState
    nodes: List[str] = field(default_factory=list)   # all known node_ids
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
