"""Extract whole-frame and spatial descriptors from pinned vision backbones."""

import os

import cv2
import numpy as np

from stideo.models import MODELS, Model, model_file

NORMALIZATION = {
    "dino": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "siglip": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "mobilevit": ((0, 0, 0), (1, 1, 1)),
}
# These tensor names belong to the immutable exports in the registry.
OUTPUT = {
    "dino": "last_hidden_state",
    "mobilevit": "/mobilevit/conv_1x1_exp/activation/Mul_output_0",
}


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class VisualEncoder:
    def __init__(self, spec: Model, net: cv2.dnn.Net) -> None:
        self.spec = spec
        self.net = net
        self.mean, self.std = (np.asarray(x, dtype=np.float32) for x in NORMALIZATION[spec.family])

    @classmethod
    def load(cls, name: str, *, offline: bool) -> VisualEncoder:
        cv2.setNumThreads(min(4, os.cpu_count() or 1))
        return cls(MODELS[name], cv2.dnn.readNetFromONNX(str(model_file(name, offline=offline))))

    def semantic(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        grid, side = self.spec.grid, self.spec.side
        patch = side // grid
        rows, cols = max(1, round(grid * h / max(w, h))), max(1, round(grid * w / max(w, h)))
        y, x = (grid - rows) // 2, (grid - cols) // 2
        scaled = cv2.resize(rgb, (cols * patch, rows * patch), interpolation=cv2.INTER_AREA)
        if self.spec.family == "mobilevit":
            scaled = scaled[:, :, ::-1]  # The MobileViT feature extractor uses BGR.
        canvas = np.zeros((side, side, 3), dtype=np.float32)
        canvas[y * patch : (y + rows) * patch, x * patch : (x + cols) * patch] = (
            scaled.astype(np.float32) / 255 - self.mean
        ) / self.std
        self.net.setInput(np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), "pixel_values")
        if self.spec.family == "clip":
            # This export exposes the contrastively trained projection as its sole public output.
            return np.sqrt(0.81) * unit(self.net.forward("image_embeds")[0])
        if self.spec.family == "siglip":
            tokens, pooled = self.net.forward(["last_hidden_state", "pooler_output"])
            spatial = tokens[0].reshape(grid, grid, -1)[y : y + rows, x : x + cols]
            global_vector = pooled[0]
        elif self.spec.family == "mobilevit":
            output = self.net.forward(OUTPUT["mobilevit"])[0]
            spatial = output.transpose(1, 2, 0)[y : y + rows, x : x + cols]
            global_vector = spatial.mean(axis=(0, 1))
        else:
            output = self.net.forward(OUTPUT["dino"])[0]
            spatial = output[1:].reshape(grid, grid, -1)[y : y + rows, x : x + cols]
            global_vector = output[0]
        patches = unit(spatial)
        regions = [
            unit(
                patches[
                    i * rows // 2 : ((i + 1) * rows + 1) // 2, j * cols // 2 : ((j + 1) * cols + 1) // 2
                ].mean(axis=(0, 1))
            )
            for i in range(2)
            for j in range(2)
        ]
        return np.concatenate(
            [
                np.sqrt(0.46) * unit(global_vector),
                np.sqrt(0.17) * unit(patches.mean(axis=(0, 1))),
                np.sqrt(0.18 / 4) * np.asarray(regions).ravel(),
            ]
        )
