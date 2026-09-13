"""Generate Project 1 overview assets without changing the source TIFFs."""

from pathlib import Path

import numpy as np
from PIL import Image


ASSETS = Path(__file__).resolve().parents[1] / "assets" / "proj01"
WEB = ASSETS / "web"


def save_preview(image, name, bounds):
    preview = image.copy()
    preview.thumbnail(bounds, Image.Resampling.LANCZOS)
    preview.save(WEB / name, quality=90, optimize=True)


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    with Image.open(ASSETS / "input.tif") as source:
        if source.mode.startswith("I;16"):
            raw = Image.fromarray((np.asarray(source) / 257).round().astype("uint8"))
        else:
            raw = source.convert("L")
        raw.save(WEB / "overview-raw-full.jpg", quality=95, optimize=True)
        save_preview(raw, "overview-raw.jpg", (600, 1200))
        band_height = raw.height // 3
        for index, channel in enumerate("bgr"):
            band = raw.crop((0, index * band_height, raw.width, (index + 1) * band_height))
            save_preview(band, f"pipeline-{channel}.jpg", (240, 240))

    with Image.open(ASSETS / "output.tif") as source:
        output = source.convert("RGB")
        output.save(WEB / "overview-output-full.jpg", quality=95, optimize=True)
        save_preview(output, "overview-output.jpg", (1000, 1000))
        for channel, band in zip("rgb", output.split()):
            save_preview(band, f"pipeline-aligned-{channel}.jpg", (240, 240))


if __name__ == "__main__":
    main()
