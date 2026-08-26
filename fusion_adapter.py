from datetime import datetime

from fusion_engine import FusionEngine, Detection


class SentryFusionAdapter:

    def __init__(self, fusion_engine: FusionEngine):
        self.fusion_engine = fusion_engine

    def process_sentry_result(
        self,
        node_id: str,
        lat: float,
        lon: float,
        result: dict,
        timestamp: datetime | None = None,
    ):
        """
        SentryDetector ke result ko FusionEngine detections
        mein convert karta hai.
        """

        if not result:
            return []

        if timestamp is None:
            timestamp = datetime.now()

        entities = []

        try:
            confidence = float(result["confidence"])
            person_count = int(result["person_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("sentry result needs numeric confidence and person_count") from exc

        # SentryDetector currently person-only hai.
        # person_count ke according individual detections create kar rahe hain.
        if person_count < 0:
            raise ValueError("person_count must be non-negative")

        for _ in range(person_count):

            detection = Detection(
                node_id=node_id,
                lat=lat,
                lon=lon,
                confidence=confidence,
                class_name="person",
                timestamp=timestamp,
            )

            entity = self.fusion_engine.process_detection(detection)

            entities.append(entity)

        return entities