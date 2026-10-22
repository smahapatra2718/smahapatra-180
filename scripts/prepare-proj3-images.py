"""Collect Project 3 images and generate their web-sized derivatives.

Originals are copied unchanged from the two submission folders into
assets/proj03/a/ and assets/proj03/b/. The notebooks save their figures as
matplotlib panels; Part A's are cut back into single tiles (assets/proj03/a/tiles/)
so the page can lay them out in HTML, and Part B's sample grids are cropped to
the grid itself, dropping the plot title. Every shown image gets a derivative in
assets/proj03/web/, named after its path with "/" replaced by "--".
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

CS180 = Path.home() / "cs180"
PART_A = CS180 / "proj3_a_submission"
PART_B = CS180 / "proj3_b_submission"
RUN_A = PART_A / "results" / "parta_updated" / "20261012-140020-182b40"
# The 3.1 web image was redone with a painting in an editing-only run; every
# other output in that run is byte-identical to RUN_A.
RUN_A_FROM = {"edit_web_image": PART_A / "results" / "parta_updated" / "20261021-194802-697d17"}
RUN_B = PART_B / "results" / "partb" / "all"
# Bells & whistles: a wider, deeper CIFAR-10 UNet trained with the same recipe.
RUN_B_DEEP = PART_B / "results" / "partb" / "cifar_deep"
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "proj03"
WEB = ASSETS / "web"

LONG_EDGE = 1200
WIDE_LONG_EDGE = 2000

A_INPUTS = ["mcway_falls.jpg", "plant.jpeg", "campanile.jpg", "painting_espresso.png", "mike.png", "squidward.png"]

# Figures whose panels can hold large white areas borrow the panel boxes of a
# figure with the same layout and full-bleed content.
LAYOUT_FROM = {
    "edit_drawing_1": "edit_campanile", "edit_drawing_2": "edit_campanile",
    "edit_web_image": "edit_campanile",
}
SLICED = [
    "training_inputs", "blur_0.25", "blur_0.5", "blur_0.75",
    "one_step_0.25", "one_step_0.5", "one_step_0.75",
    "velocity_0.25", "velocity_0.50", "velocity_0.75",
    "trajectory", "comparison",
    "edit_campanile", "edit_my_image_1", "edit_my_image_2",
    "edit_web_image", "edit_drawing_1", "edit_drawing_2",
    "text_edit_campanile", "text_edit_my_image_1", "text_edit_my_image_2",
]
A_SINGLES = ([f"conditional_{i}" for i in range(5)] + [f"cfg_{i}" for i in range(5)]
             + ["anagram_0", "anagram_1", "hybrid_0", "hybrid_1", "velocity_original"])


def web_name(rel):
    return rel.rsplit(".", 1)[0].replace("/", "--") + ".jpg"


def flatten(im):
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.getchannel("A"))
        return flat
    return im.convert("RGB") if im.mode not in ("RGB", "L") else im


def derive(src, rel, keep_pixels=False):
    with Image.open(src) as im:
        out = flatten(im)
        if keep_pixels:
            # MNIST grids: nearest-neighbour only, never smooth the digits.
            out = out.copy()
        else:
            edge = WIDE_LONG_EDGE if out.width > 2.5 * out.height else LONG_EDGE
            out = out.copy()
            out.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        out.save(WEB / web_name(rel), quality=88, optimize=True, progressive=True)


def runs(profile, min_len):
    spans, start = [], None
    for i, on in enumerate(list(profile) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_len:
                spans.append((start, i))
            start = None
    return spans


def panel_boxes(path):
    """Boxes of the image axes in a matplotlib figure, row-major."""
    with Image.open(path) as im:
        a = np.asarray(flatten(im).convert("L"), dtype=np.int16)
    ink = a < 250
    boxes = []
    for x0, x1 in runs(ink.mean(axis=0) > 0.45, 60):
        for y0, y1 in runs(ink[:, x0:x1].mean(axis=1) > 0.6, 60):
            boxes.append((x0, y0, x1, y1))
    return sorted(boxes, key=lambda b: (b[1] // 50, b[0]))


def run_a(name):
    return RUN_A_FROM.get(name, RUN_A) / f"{name}.png"


def slice_figure(name):
    src = run_a(name)
    boxes = panel_boxes(RUN_A / f"{LAYOUT_FROM.get(name, name)}.png")
    with Image.open(src) as im:
        im = flatten(im)
        for i, box in enumerate(boxes):
            rel = f"a/tiles/{name}_{i}.png"
            (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
            im.crop(box).save(ASSETS / rel)
            derive(ASSETS / rel, rel)
    return len(boxes)


def crop_grid(name, run=RUN_B):
    """Crop a Part B sample grid to its dark image area."""
    with Image.open(run / f"{name}.png") as im:
        im = flatten(im)
        a = np.asarray(im.convert("L"))
        dark = a < 200
        cols = np.where(dark.mean(axis=0) > 0.5)[0]
        rows = np.where(dark[:, cols[0]:cols[-1] + 1].mean(axis=1) > 0.5)[0]
        rel = f"b/grids/{name}.png"
        (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
        im.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)).save(ASSETS / rel)
        derive(ASSETS / rel, rel, keep_pixels=True)


def derive_hybrid_far():
    # Stand-in for viewing from a distance: shrinking to 32 px keeps only the
    # low frequencies; scaling back up makes that visible at a readable size.
    for i in range(2):
        rel = f"a/derived/hybrid_{i}_far.png"
        (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
        with Image.open(RUN_A / f"hybrid_{i}.png") as im:
            small = flatten(im).resize((32, 32), Image.Resampling.LANCZOS)
        small.resize((512, 512), Image.Resampling.BICUBIC).save(ASSETS / rel)
        derive(ASSETS / rel, rel)


def derive_unet_io():
    # The 1.1 diagram's input and output: one test digit from the 1.2.1 epoch-5
    # grid (rows: clean, noisy, denoised). make_grid pads 2 px around 28 px cells.
    with Image.open(ASSETS / "b" / "grids" / "1.2.1_denoise_epoch5.png") as im:
        im = im.convert("L")
        scale = im.width / (8 * 30 + 2)
        for name, row in (("unet_in", 1), ("unet_out", 2)):
            x0, y0 = 2 * scale, (2 + row * 30) * scale
            cell = im.crop((round(x0), round(y0), round(x0 + 28 * scale), round(y0 + 28 * scale)))
            rel = f"b/derived/{name}.png"
            (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
            cell.resize((112, 112), Image.Resampling.NEAREST).save(ASSETS / rel)
            derive(ASSETS / rel, rel, keep_pixels=True)

        # 2.1 diagram input: the same test digit as x_t = (1 - t) x_0 + t x_1 at
        # t = 0.5, an illustration of the formula (no model involved).
        x0, y0 = 2 * scale, 2 * scale
        clean = im.crop((round(x0), round(y0), round(x0 + 28 * scale), round(y0 + 28 * scale)))
        x1 = np.asarray(clean.resize((28, 28), Image.Resampling.BOX), dtype=float) / 255
        eps = np.random.default_rng(0).standard_normal((28, 28))
        xt = np.clip(0.5 * eps + 0.5 * x1, 0, 1)
        rel = "b/derived/tunet_in.png"
        Image.fromarray((xt * 255).round().astype("uint8")).resize((112, 112), Image.Resampling.NEAREST).save(ASSETS / rel)
        derive(ASSETS / rel, rel, keep_pixels=True)


def derive_dunet_in():
    # Input for the wider/deeper CIFAR diagram: CIFAR-10 test image #3 (an
    # airplane) as x_t = (1 - t) x_0 + t x_1 at t = 0.5, for illustration only.
    import torch
    data = torch.load(PART_B / "data" / "cifar10_test.pt", weights_only=True)
    x1 = data["images"][3].permute(1, 2, 0).numpy().astype(float) / 255
    eps = np.random.default_rng(0).standard_normal(x1.shape)
    xt = np.clip(0.5 * eps + 0.5 * x1, 0, 1)
    rel = "b/derived/dunet_in.png"
    (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((xt * 255).round().astype("uint8")).resize((128, 128), Image.Resampling.NEAREST).save(ASSETS / rel)
    derive(ASSETS / rel, rel, keep_pixels=True)


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    for name in A_INPUTS:
        rel = f"a/src/{name}"
        (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PART_A / "assets" / name, ASSETS / rel)
        derive(ASSETS / rel, rel)
    for p in sorted(RUN_A.glob("*.png")):
        p = run_a(p.stem)
        rel = f"a/out/{p.name}"
        (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, ASSETS / rel)
        if p.stem in A_SINGLES:
            derive(ASSETS / rel, rel)
    for name in SLICED:
        print(f"{name}: {slice_figure(name)} panels")
    derive_hybrid_far()
    for p in sorted(RUN_B.glob("*.png")):
        rel = f"b/out/{p.name}"
        (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, ASSETS / rel)
        if p.stem.endswith("loss"):
            derive(ASSETS / rel, rel)
        else:
            crop_grid(p.stem)
    derive_unet_io()
    for p in sorted(RUN_B_DEEP.glob("*.png")):
        rel = f"b/out/{p.name}"
        shutil.copy2(p, ASSETS / rel)
        if p.stem.endswith("loss"):
            derive(ASSETS / rel, rel)
        else:
            crop_grid(p.stem, RUN_B_DEEP)
    derive_dunet_in()


if __name__ == "__main__":
    main()
