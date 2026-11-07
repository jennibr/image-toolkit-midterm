"""Feature 1: image loading and basic metadata parsing."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "bmp"]
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP"}
MAX_SIDE = 2000  # larger images are downscaled to keep the UI responsive


class ImageLoadError(Exception):
    """Raised when the uploaded file cannot be used as an image."""


@dataclass
class ImageInfo:
    filename: str
    fmt: str
    width: int
    height: int
    original_mode: str
    channels: int
    bit_depth: int
    file_size_bytes: int
    resized_from: tuple[int, int] | None = None

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def file_size_human(self) -> str:
        return human_size(self.file_size_bytes)


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{n} B"


_MODE_BITS = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "CMYK": 8, "LA": 8,
              "I;16": 16, "I": 32, "F": 32}


def load_image(file_bytes: bytes, filename: str = "image") -> tuple[np.ndarray, ImageInfo]:
    """Decode bytes to an RGB uint8 array (H, W, 3) plus metadata."""
    if not file_bytes:
        raise ImageLoadError("The file is empty.")
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(f"Cannot decode image: {exc}") from exc

    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ImageLoadError(f"Unsupported format '{fmt}'. Use PNG, JPG/JPEG or BMP.")

    img = ImageOps.exif_transpose(img)  # respect camera orientation
    mode = img.mode
    channels = len(img.getbands())

    # Flatten transparency onto white so RGB math stays well defined
    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        img = img.convert("RGB")

    resized_from = None
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        resized_from = (w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)
    info = ImageInfo(
        filename=filename,
        fmt=fmt,
        width=arr.shape[1],
        height=arr.shape[0],
        original_mode=mode,
        channels=channels,
        bit_depth=_MODE_BITS.get(mode, 8),
        file_size_bytes=len(file_bytes),
        resized_from=resized_from,
    )
    return arr, info