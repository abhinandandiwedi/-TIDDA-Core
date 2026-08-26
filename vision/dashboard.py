def frame_payload(result: dict) -> dict:
    """Create JSON-ready WebSocket payload data without coupling vision to transport."""
    cloud = result.get("point_cloud")
    return {"type": "vision_frame", "frame_index": result["frame"].index,
            "timestamp": result["frame"].timestamp.isoformat(),
            "quality": {"valid": result["quality"].valid, "brightness": result["quality"].brightness,
                        "sharpness": result["quality"].sharpness},
            "detections": [{"class_name": item.class_name, "confidence": item.confidence,
                            "track_id": item.track_id, "bbox": {"x1": item.bbox.x1, "y1": item.bbox.y1,
                            "x2": item.bbox.x2, "y2": item.bbox.y2}} for item in result["detections"]],
            "point_count": len(cloud.points) if cloud is not None else 0,
            "entities": [entity.to_dict() for entity in result.get("entities", [])]}
