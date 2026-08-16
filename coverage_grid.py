# ══════════════════════════════════════════════════════════════════
#  📐 TIDDA COVERAGE GRID — GPS → cell-based area coverage
#  Converts streaming GPS points into a grid of "surveyed" cells.
#  Completely isolated from drone physics — used by mobile nodes only.
# ══════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

# ── Geo constant (equirectangular approximation) ─────────────────
METERS_PER_DEG_LAT: float = 111_320.0


class CoverageGrid:
    """Track which geographic cells have been visited by any mobile node.

    The operating area is divided into a uniform grid of square cells.
    As GPS positions arrive, the containing cell is marked as covered.
    Only *newly* covered cells are returned (delta), so the caller can
    broadcast incremental updates without resending the entire grid.

    Cell coordinates are integer indices (cx, cy) relative to an origin
    point.  The origin is set on first use or via constructor args.

    Parameters
    ----------
    cell_size_m : float
        Side length of each grid cell in meters (default 10m).
    origin_lat : float | None
        Latitude of the grid origin (bottom-left corner).
        If None, the first GPS point received becomes the origin.
    origin_lon : float | None
        Longitude of the grid origin.
    """

    def __init__(
        self,
        cell_size_m: float = 10.0,
        origin_lat: float | None = None,
        origin_lon: float | None = None,
    ) -> None:
        self.cell_size_m = cell_size_m
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._origin_set = origin_lat is not None and origin_lon is not None

        # Set of covered cell keys: (cx, cy)
        self._covered: Set[Tuple[int, int]] = set()

        # Precompute degrees-per-cell for lat (lon depends on latitude)
        self._deg_per_cell_lat = cell_size_m / METERS_PER_DEG_LAT

    # ── Core API ─────────────────────────────────────────────────

    def record_position(self, lat: float, lon: float) -> List[Tuple[int, int, float, float]]:
        """Record a GPS position and return any newly covered cells.

        Returns a list of (cx, cy, center_lat, center_lon) tuples for
        cells that were just covered for the first time.  An empty list
        means the position fell in an already-covered cell.
        """
        # Auto-set origin on first call if not provided
        if not self._origin_set:
            self._origin_lat = lat
            self._origin_lon = lon
            self._origin_set = True

        cx, cy = self._latlng_to_cell(lat, lon)

        if (cx, cy) in self._covered:
            return []

        self._covered.add((cx, cy))
        center_lat, center_lon = self._cell_to_center(cx, cy)
        return [(cx, cy, center_lat, center_lon)]

    def total_covered(self) -> int:
        """Number of unique cells covered so far."""
        return len(self._covered)

    def all_cells(self) -> List[Tuple[int, int, float, float]]:
        """Return all covered cells as (cx, cy, center_lat, center_lon).

        Used for full-grid sync when a new dashboard client connects.
        """
        return [
            (cx, cy, *self._cell_to_center(cx, cy))
            for cx, cy in self._covered
        ]

    def reset(self) -> None:
        """Clear all coverage data (new session)."""
        self._covered.clear()

    @property
    def cell_size(self) -> float:
        """Cell size in meters."""
        return self.cell_size_m

    @property
    def origin(self) -> Tuple[float | None, float | None]:
        """Return (origin_lat, origin_lon) or (None, None) if unset."""
        return (self._origin_lat, self._origin_lon)

    # ── Internal geometry ────────────────────────────────────────

    def _deg_per_cell_lon(self) -> float:
        """Degrees of longitude per cell, corrected for latitude."""
        if self._origin_lat is None:
            return self._deg_per_cell_lat  # fallback (equator)
        cos_lat = math.cos(math.radians(self._origin_lat))
        if cos_lat < 1e-9:
            cos_lat = 1e-9  # pole guard
        return self.cell_size_m / (METERS_PER_DEG_LAT * cos_lat)

    def _latlng_to_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert a lat/lon to integer cell indices (cx, cy)."""
        assert self._origin_lat is not None and self._origin_lon is not None
        cy = int(math.floor((lat - self._origin_lat) / self._deg_per_cell_lat))
        cx = int(math.floor((lon - self._origin_lon) / self._deg_per_cell_lon()))
        return (cx, cy)

    def _cell_to_center(self, cx: int, cy: int) -> Tuple[float, float]:
        """Convert cell indices back to the center lat/lon of that cell."""
        assert self._origin_lat is not None and self._origin_lon is not None
        center_lat = self._origin_lat + (cy + 0.5) * self._deg_per_cell_lat
        center_lon = self._origin_lon + (cx + 0.5) * self._deg_per_cell_lon()
        return (center_lat, center_lon)
