from pathlib import Path


class DepthEstimator:
    """Replaceable monocular depth adapter using Hugging Face DPT."""

    def __init__(self, model_name: str = "Intel/dpt-hybrid-midas") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def estimate(self, image_path: Path):
        import numpy as np
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth -= depth.min()
        maximum = depth.max()
        return depth / maximum if maximum > 0 else depth
