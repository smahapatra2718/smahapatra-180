"""Collect Project 2 images and generate their web-sized derivatives.

Originals are copied unchanged from the submission folder into assets/proj02/
(out/ for results, src/ for inputs). Every one of them gets a web derivative in
assets/proj02/web/, named after its path with "/" replaced by "--", e.g.
  out/1_2/thresholds/cameraman_binarized_t0.1.jpg
    -> web/out--1_2--thresholds--cameraman_binarized_t0.1.jpg
"""

import shutil
from pathlib import Path

from PIL import Image

SUBMISSION = Path.home() / "cs180" / "proj2_submission"
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "proj02"
WEB = ASSETS / "web"

# Inputs the page shows. Leftovers from dropped blends are not copied.
SOURCES = [
    "selfie.jpg", "cameraman.png", "taj.jpg", "oakland.jpg",
    "derek.jpg", "nutmeg.jpg", "sunflower.jpg", "lion.jpg",
    "day_aligned.jpg", "night_aligned.jpg",
    "apple.jpeg", "orange.jpeg", "cat.jpeg", "tiger.jpg", "cat_body_mask.png",
]

LONG_EDGE = 1200       # column is 800px; this covers 1.5x displays
WIDE_LONG_EDGE = 2000  # panoramas span the column at a fraction of their height
KERNEL_PREFIX = ("out/1_3/gaussian_kernel", "out/1_3/dogx", "out/1_3/dogy")


def web_name(rel):
    return rel.rsplit(".", 1)[0].replace("/", "--") + ".jpg"


def derive(src, rel):
    with Image.open(src) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, (255, 255, 255))
            flat.paste(im, mask=im.getchannel("A"))
            im = flat
        elif im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if rel.startswith(KERNEL_PREFIX):
            # Already block-upscaled; resampling would smear the cells.
            out = im.copy()
        else:
            edge = WIDE_LONG_EDGE if im.width > 2.5 * im.height else LONG_EDGE
            out = im.copy()
            out.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        out.save(WEB / web_name(rel), quality=86, optimize=True, progressive=True)


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    pairs = [(SUBMISSION / "assets" / name, f"src/{name}") for name in SOURCES]
    pairs += [(p, "out/" + p.relative_to(SUBMISSION / "out").as_posix())
              for p in sorted((SUBMISSION / "out").rglob("*"))
              if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
              and "figures" not in p.parts]
    for src, rel in pairs:
        dest = ASSETS / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        derive(dest, rel)
    derive_square_dog()
    derive_oraple_mask_stack()
    print(f"{len(pairs) + 20} images -> {WEB.relative_to(ASSETS.parents[1])}")


def derive_oraple_mask_stack():
    # main.py blurs the oraple mask into a Gaussian stack but never saves it.
    # Rebuild it (and 1 - mask) with the submission's own gaussian_stack and the
    # oraple's settings (left-half step mask, sigma 5, 6 levels), without
    # writing anything back into the submission folder.
    import sys
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SUBMISSION))
    import numpy as np
    import methods as m

    with Image.open(ASSETS / "src" / "apple.jpeg") as apple:
        h, w = apple.height, apple.width
    mask = np.zeros((h, w, 3))
    mask[:, : w // 2] = 1
    stack = m.gaussian_stack(mask, 5, 6)
    (ASSETS / "derived").mkdir(parents=True, exist_ok=True)
    for i, level in enumerate(stack):
        for name, values in (("mask", level[..., 0]), ("mask_inv", 1 - level[..., 0])):
            rel = f"derived/oraple_{name}_gaussian_{i}.png"
            Image.fromarray((np.clip(values, 0, 1) * 255).round().astype("uint8")).save(ASSETS / rel)
            derive(ASSETS / rel, rel)

    # Every blended level (main.py only saves levels 0, 2 and 4). Band-pass levels
    # use the same signed display mapping as save_blend_figure; the last level is
    # a low-pass image and is shown as-is, as the Laplacian stacks do.
    import skimage as sk
    import skimage.io as skio
    apple = sk.img_as_float(skio.imread(ASSETS / "src" / "apple.jpeg"))
    orange = sk.img_as_float(skio.imread(ASSETS / "src" / "orange.jpeg"))
    result = m.blend(apple, orange, mask, 5, 6)
    for i, level in enumerate(result["blended_stack"]):
        shown = level if i == len(result["blended_stack"]) - 1 else m.to_display(level)
        rel = f"derived/oraple_blended_level_{i}.png"
        Image.fromarray((np.clip(shown, 0, 1) * 255).round().astype("uint8")).save(ASSETS / rel)
        derive(ASSETS / rel, rel)



def derive_square_dog():
    # DoG x is 9x11 cells and DoG y is 11x9. Pad each with zero-valued cells on
    # its short sides so both display as 11x11; zero padding around the centre
    # leaves the filter unchanged. In the saved images, 0 maps to mid-gray.
    cell, zero = 30, 128
    for name in ("dogx", "dogy"):
        with Image.open(ASSETS / "out" / "1_3" / f"{name}.jpg") as im:
            im = im.convert("L")
            side = max(im.size)
            padded = Image.new("L", (side, side), zero)
            padded.paste(im, ((side - im.width) // 2 // cell * cell, (side - im.height) // 2 // cell * cell))
        rel = f"derived/{name}_square.png"
        (ASSETS / "derived").mkdir(parents=True, exist_ok=True)
        padded.save(ASSETS / rel)
        padded.save(WEB / web_name(rel), quality=95)




if __name__ == "__main__":
    main()
