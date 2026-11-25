"""Original procedural stages and moving three-dimensional objects for ranking tests."""

import cv2
import numpy as np

WIDTH, HEIGHT = 192, 144
SUBJECTS = ("chair", "table", "biplane", "tugboat")
LOCATIONS = ("bookshop", "platform", "greenhouse", "harbor")
PALETTES = ((178, 80, 67), (69, 116, 158), (132, 152, 87), (165, 117, 155))
# Each solid has a centre, its three dimensions and a material index.
type Solid = tuple[float, float, float, float, float, float, int]
LEGS: list[Solid] = [(x, -0.38, z, 0.16, 0.95, 0.16, 1) for x in (-0.51, 0.51) for z in (-0.43, 0.43)]
SOLIDS: dict[str, list[Solid]] = {
    "chair": [*LEGS, (0, 0.14, 0, 1.25, 0.18, 1.05, 0), (0, 0.66, 0.45, 1.25, 0.94, 0.14, 0)],
    "table": [*LEGS, (0, 0.21, 0, 1.83, 0.20, 1.55, 0), (0, 0.38, 0, 0.32, 0.14, 0.35, 2)],
    "biplane": [
        (0, 0, 0, 1.65, 0.32, 0.37, 0),
        (0.13, 0.44, 0, 0.50, 0.10, 2.12, 0),
        (0.13, -0.11, 0, 0.55, 0.11, 2.1, 2),
        (-0.7, 0.26, 0, 0.30, 0.51, 0.11, 0),
        *[(0.15, 0.15, z, 0.07, 0.53, 0.07, 1) for z in (-0.76, 0.76)],
    ],
    "tugboat": [
        (0, -0.43, 0, 2.1, 0.46, 0.85, 0),
        (0, -0.15, 0, 1.88, 0.12, 0.75, 2),
        (0.1, 0.14, 0, 0.83, 0.50, 0.55, 2),
        (-0.28, 0.51, 0, 0.19, 0.84, 0.19, 1),
        (0.16, 0.44, 0, 0.91, 0.09, 0.70, 0),
    ],
}
# Independently constructed silhouettes reserved for the holdout.
HOLDOUT_SOLIDS: dict[str, list[Solid]] = {
    "chair": [
        *LEGS,
        (0, 0.14, 0, 1.3, 0.15, 1.12, 2),
        *[(x, 0.6, 0.47, 0.14, 1.12, 0.15, 0) for x in (-0.52, 0, 0.52)],
        (0, 1.06, 0.47, 1.17, 0.19, 0.16, 0),
        *[(x, 0.49, 0, 0.12, 0.15, 0.97, 0) for x in (-0.58, 0.58)],
    ],
    "table": [
        (0, 0.27, 0, 1.85, 0.13, 1.43, 0),
        (0, -0.27, 0, 0.25, 1.0, 0.27, 1),
        (0, -0.73, 0, 1.17, 0.13, 0.22, 2),
        (0, -0.73, 0, 0.23, 0.14, 1.19, 2),
    ],
    "biplane": [
        (0, 0, 0, 2.1, 0.33, 0.34, 0),
        (0.08, 0.36, 0, 0.59, 0.09, 1.93, 2),
        (0.09, -0.13, 0, 0.51, 0.1, 2.26, 0),
        (-0.83, 0.04, 0, 0.41, 0.10, 0.88, 1),
        (-0.90, 0.24, 0, 0.21, 0.35, 0.12, 0),
        *[(0.12, 0.16, z, 0.09, 0.54, 0.08, 1) for z in (-0.76, 0.76)],
    ],
    "tugboat": [
        (0, -0.47, 0, 1.95, 0.38, 1.03, 1),
        (0, -0.23, 0, 2.21, 0.16, 0.98, 0),
        (-0.15, 0.03, 0, 1.02, 0.43, 0.65, 2),
        (-0.31, 0.34, 0, 0.48, 0.21, 0.56, 0),
        (0.39, 0.46, 0, 0.13, 1.12, 0.13, 1),
    ],
}
SIGNS = np.array(
    [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    dtype=float,
)
FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5), (3, 2, 6, 7), (0, 1, 5, 4))


def studio(y: int, accent: tuple[int, int, int]) -> np.ndarray:
    sky = np.linspace(0.75, 1.0, HEIGHT)[:, None, None] * (np.array(accent) * 0.24 + 155)
    rgb = np.repeat(sky, WIDTH, axis=1).astype(np.uint8)
    rgb[y:] = (rgb[y:] * 0.76).astype(np.uint8)
    return rgb


def object_frame(label: str, angle: float, variant: int, seed: int, *, holdout: bool = False) -> np.ndarray:
    tone = PALETTES[(seed + variant) % len(PALETTES)]
    rgb = studio(96 + seed % 6, PALETTES[(seed + 2 * variant) % len(PALETTES)])
    ct, st = np.cos(angle), np.sin(angle)
    elevation = 0.27 + 0.10 * variant
    ce, se = np.cos(elevation), np.sin(elevation)
    rotation = np.array([[ct, 0, st], [-st * se, ce, ct * se], [-st * ce, -se, ct * ce]])
    center = np.array([WIDTH / 2 + (seed % 15 - 7), HEIGHT * 0.53])
    cv2.ellipse(rgb, (round(center[0]), 105), (46, 8), 0, 0, 360, (116, 122, 123), -1)
    materials = np.array([tone, (63, 69, 74), (218, 207, 174)])
    faces = []
    for x, y, z, w, h, d, material in (HOLDOUT_SOLIDS if holdout else SOLIDS)[label]:
        world = (SIGNS * (w, h, d) * 0.5 + (x, y, z)) * (1 + seed % 5 / 28, 1 + seed % 7 / 30, 1)
        xyz = world @ rotation.T
        pixels = np.rint(center + xyz[:, :2] * (39, -39)).astype(np.int32)
        for face, shade in zip(FACES, (0.78, 0.86, 0.65, 0.93, 1.05, 0.62), strict=True):
            indices = list(face)
            color = tuple(int(v) for v in np.clip(materials[material] * shade, 0, 255))
            faces.append((xyz[indices, 2].mean(), pixels[indices], color))
    for _, polygon, color in sorted(faces, key=lambda item: item[0]):
        cv2.fillConvexPoly(rgb, polygon, color, cv2.LINE_AA)
    return rgb


def harbor(
    picture: np.ndarray, colors: list[list[int]], dark: tuple[int, ...], pale: tuple[int, ...]
) -> None:
    picture[92:] = (111, 142, 151)
    for x in range(-10, 270, 42):
        cv2.line(picture, (x, 81), (x, 150), dark, 5)
        cv2.line(picture, (x, 104), (x + 42, 96), pale, 3)
    for x in (48, 128, 198):
        cv2.fillConvexPoly(picture, np.int32([[x, 84], [x, 41], [x - 21, 84]]), colors[x % 4])


def location_backdrop(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picture = np.empty((180, 256, 3), np.uint8)
    picture[:] = (181, 181, 171)
    picture[112:] = (112, 115, 110)
    dark, pale = (63, 68, 72), (217, 211, 190)
    colors = rng.permutation(np.asarray(PALETTES)).tolist()
    if label == "bookshop":
        for y in (48, 90, 132):
            for x in range(-5, 255, 12):
                height = rng.integers(12, 32).item()
                cv2.rectangle(picture, (x, y - height), (x + 8, y - 3), colors[(x + y) % 4], -1)
            cv2.rectangle(picture, (0, y), (255, y + 5), dark, -1)
    elif label == "platform":
        for x in (10, 52, 97, 146, 208):
            cv2.line(picture, (127, 104), (x * 3 - 128, 180), dark, 4)
            cv2.line(picture, (x, 28), (x, 104), dark, 4)
        cv2.fillConvexPoly(picture, np.int32([[0, 12], [250, 33], [174, 46], [0, 32]]), colors[1])
    elif label == "greenhouse":
        for x in range(-50, 270, 32):
            cv2.line(picture, (128, 12), (x, 100), pale, 3)
            cv2.line(picture, (x, 97), (x, 128), dark, 3)
        for x in range(5, 255, 18):
            height = rng.integers(14, 39).item()
            cv2.ellipse(
                picture, (x, 118 - height // 2), (6, height // 2), x % 30 - 15, 0, 360, colors[x % 4], -1
            )
    else:
        harbor(picture, colors, dark, pale)
    return picture


def location_frame(background: np.ndarray, tick: int, variant: int, seed: int) -> np.ndarray:
    x = (seed % 19 + 9 * variant + tick // 2) % 58
    y = (seed + 5 * variant) % 32
    image = background[y : y + HEIGHT, x : x + WIDTH].copy()
    # This prominent, moving foreground recurs at every location.
    center = (56 + variant * 29 + tick // 2, 90)
    cv2.ellipse(image, center, (17, 26), 8, 0, 360, PALETTES[variant], -1, cv2.LINE_AA)
    cv2.circle(image, (center[0] + 2, 60), 11, (150, 124, 102), -1, cv2.LINE_AA)
    return image
