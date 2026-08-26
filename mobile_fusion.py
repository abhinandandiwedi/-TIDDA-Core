from datetime import datetime

from fusion_engine import FusionEngine, Detection
from mobile_node import MobileNodeRegistry


class MobileFusionService:

    def __init__(self):
        self.registry = MobileNodeRegistry()

        self.fusion = FusionEngine(
            distance_threshold_m=100.0,
            time_window_s=10.0,
            stale_after_s=15.0,
        )

    def process_sentry_result(
        self,
        node_id: str,
        sentry_result: dict,
    ):
        """
        Mobile node ke SentryDetector result ko
        FusionEngine tak pahunchata hai.
        """

        # Registered mobile node obtain karo
        node = self.registry.get_node(node_id)

        if node is None:
            raise ValueError(
                f"Mobile node '{node_id}' is not registered"
            )

        # GPS available hona chahiye
        if node.lat == 0.0 and node.lon == 0.0:
            raise ValueError(
                f"Mobile node '{node_id}' has no valid GPS position"
            )

        if not sentry_result:
            return []

        entities = []

        try:
            confidence = float(sentry_result["confidence"])
            person_count = int(sentry_result["person_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sentry result needs numeric confidence and person_count") from exc

        if person_count < 0:
            raise ValueError("person_count must be non-negative")

        timestamp = datetime.now()

        for _ in range(person_count):

            detection = Detection(
                node_id=node.node_id,
                lat=node.lat,
                lon=node.lon,
                confidence=confidence,
                class_name="person",
                timestamp=timestamp,
            )

            entity = self.fusion.process_detection(
                detection
            )

            entities.append(entity)

        return entities
        