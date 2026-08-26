from mobile_fusion import MobileFusionService


service = MobileFusionService()


# ==========================================
# REGISTER TWO MOBILE NODES
# ==========================================

service.registry.register(
    node_id="PHONE-01",
    mode="sentry"
)

service.registry.register(
    node_id="PHONE-02",
    mode="sentry"
)


# ==========================================
# GPS TELEMETRY
# Both phones are very close
# ==========================================

service.registry.update_telemetry(
    "PHONE-01",
    {
        "lat": 26.846700,
        "lon": 80.946200,
        "camera_active": True,
        "mode": "sentry",
        "status": "ACTIVE",
    }
)

service.registry.update_telemetry(
    "PHONE-02",
    {
        "lat": 26.846750,
        "lon": 80.946250,
        "camera_active": True,
        "mode": "sentry",
        "status": "ACTIVE",
    }
)


# ==========================================
# PHONE 1 DETECTS PERSON
# ==========================================

result_1 = {
    "confidence": 0.82,
    "person_count": 1,
    "bboxes": [
        [100.0, 80.0, 220.0, 300.0]
    ],
}

entities_1 = service.process_sentry_result(
    "PHONE-01",
    result_1
)


# ==========================================
# PHONE 2 DETECTS PERSON
# Same area + same time window
# ==========================================

result_2 = {
    "confidence": 0.91,
    "person_count": 1,
    "bboxes": [
        [110.0, 85.0, 225.0, 305.0]
    ],
}

entities_2 = service.process_sentry_result(
    "PHONE-02",
    result_2
)


# ==========================================
# RESULT
# ==========================================

print("\n===== MULTI-NODE FUSION TEST =====")

for entity_id, entity in service.fusion.entities.items():

    print(
        entity_id,
        "| class:", entity.class_name,
        "| detections:", len(entity.detections),
        "| confidence:", entity.confidence,
        "| threat:", round(entity.threat_score, 2)
    )

    print("  Sources:")

    for detection in entity.detections:
        print(
            "   -",
            detection.node_id,
            "| confidence:",
            detection.confidence
        )