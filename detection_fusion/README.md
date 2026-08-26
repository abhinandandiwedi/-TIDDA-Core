# Multi-Sensor Fusion Engine

A modular Python 3.10+ core for turning camera, thermal, radar, LiDAR, MAVLink, MQTT, UDP, REST, or WebSocket observations into tracked entities. Network and drone adapters stay outside the package.

## Quick start

```python
from datetime import datetime, timezone
from fusion import Detection, FusionEngine

engine = FusionEngine()
entity = engine.add_detection(Detection(
    node_id="camera_01", lat=26.8467, lon=80.9462,
    confidence=0.91, class_name="vehicle",
    timestamp=datetime.now(timezone.utc),
))
print(entity.to_dict())
```

Malformed dictionaries are rejected safely and emit `INVALID_DETECTION`; valid detections return an `Entity`.

## Architecture

`FusionEngine` coordinates `Detection`, `Matcher`, `PositionFusion`, `ConfidenceFusion`, `ThreatScorer`, `EntityManager`, and `EventManager`. `Distance.py` contains the Haversine calculation. `Entity` is transport/database independent and serializes with `to_dict()` or `to_json()`.

Matching first rejects class, distance, and time-window mismatches. Candidates receive configurable distance, time, class, and confidence scores; the highest score wins. Matching is currently O(n) over active/stale entities, leaving spatial indexing as a future optimization.

Position is a confidence-weighted latitude/longitude average. Confidence uses noisy-OR, `1 - product(1-p)`, so independent supporting sensors increase confidence without exceeding 1.0. Both are isolated components and can later be replaced by a Kalman filter or another estimator.

Threat scoring combines configurable class prior, fused confidence, persistence, sensor support, and optional speed into 0.0-10.0. Levels are LOW (0-2), MEDIUM (3-5), HIGH (6-8), and CRITICAL (9-10).

## Lifecycle and events

Entities begin `NEW`, become `ACTIVE` after observation, become `STALE` after `stale_timeout_seconds`, and become `EXPIRED` after `expire_timeout_seconds`. `detect_stale_entities()` changes state without deleting records; `cleanup_expired_entities()` explicitly removes expired records.

Subscribe with `engine.subscribe_to_event(callback)`. Events include `ENTITY_CREATED`, `ENTITY_MATCHED`, `ENTITY_UPDATED`, `THREAT_CHANGED`, `ENTITY_STALE`, `ENTITY_EXPIRED`, and `INVALID_DETECTION`. History is bounded by `max_history_size`.

## Configuration and integration

Use `FusionConfig` for all thresholds, weights, aliases, threat class priors, and history limits. Feed adapters should translate their native payload into `Detection` and call `engine.add_detection()`. No database, MAVLink, or network dependency is required; persistence and transports can be added around the engine.

## Run

```powershell
python -m pytest fusion/tests -q
python -m fusion.examples.basic_usage
```

The package uses only the Python standard library. Pytest is needed only for tests.
