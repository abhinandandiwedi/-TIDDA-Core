class WorldModel:
    def __init__(self):
        self.point_cloud_url = None

    def update_point_cloud(self, url: str):
        self.point_cloud_url = url
