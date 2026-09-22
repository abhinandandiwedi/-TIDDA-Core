# Map loader utility for dashboard (served via static files in FastAPI)
def get_map_url(model_name: str) -> str:
    return f"/models/{model_name}"
