"""Collect Project 4A images and generate their web-sized derivatives.

Originals are copied unchanged from the submission folder: the photographs into
assets/proj04/src/, the A.3 and A.4 results into assets/proj04/out/, and the
clicked points, printed systems and timings into assets/proj04/data/. The A.3
rectified crops and detail crops are cut from the rectified outputs into
assets/proj04/derived/, beside a forward-warped version of one poster for
comparison. Every shown image gets a derivative in
assets/proj04/web/, named after its path with "/" replaced by "--".

Figures and values drawn from numbers rather than pixels (correspondence
markers, the rectification corners, [A | b] and H, point counts, reprojection
errors, output sizes and timings) are written into proj4.html between
<!-- gen:NAME --> and <!-- /gen:NAME --> markers, so re-clicking points and
rerunning this script keeps the page in step. Nothing outside the markers is
touched.
"""

import io
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path.home() / "cs180" / "proj_4_a_submission" / "code"
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "proj04"
WEB = ASSETS / "web"
PAGE = ROOT / "proj4.html"

LONG_EDGE = 1200
RESULT_LONG_EDGE = 1600
# main.py scales every photo so its long side is at most 2048 px; clicked points
# are in that frame.
MAX_SIDE = 2048
# Offer the "points" view only when it magnifies at least this much.
MIN_ZOOM = 1.5

# Each mosaic warps <m>_1 onto <m>_2, the reference.
MOSAICS = ["tahoe", "gateway", "vlsb"]
POSTERS = ["poster_1", "poster_2"]

# A.3 detail crops, framed on each poster's lettering: centre (u, v) and width,
# in fractions of the clicked rectangle, so the frame follows the poster when its
# aspect ratio is re-measured. Crops are 2:1 to fill the detail slider.
DETAIL = {"poster_1": (0.47, 0.45, 0.5), "poster_2": (0.37, 0.145, 0.5)}
# The full-resolution copy is pixel-replicated, as in main.py's zoom().
DETAIL_SCALE = 4
# Forward warping is shown on the poster that rectification magnifies, where
# its holes show; the other is shrunk and would barely have any.
FORWARD = ["poster_1"]

# Marker colours, in this fixed order (validated for colour-vision deficiency on
# adjacent pairs). Past eight points the order repeats; the number on each marker
# and the line joining each pair carry identity, colour only helps the eye.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def web_name(rel, suffix=""):
    return rel.rsplit(".", 1)[0].replace("/", "--") + suffix + ".jpg"


def web(rel, suffix=""):
    return f"assets/proj04/web/{web_name(rel, suffix)}"


def write_bytes(path, data):
    """Write only when the content changed, and atomically, so a page open in a
    browser never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_image(im, path, **opts):
    buf = io.BytesIO()
    im.save(buf, format="PNG" if path.suffix == ".png" else "JPEG", **opts)
    write_bytes(path, buf.getvalue())


def copy(src, rel):
    # The submission folder may be cleaned out (outputs deleted) for turn-in;
    # keep the copy already here rather than failing.
    if not src.exists() and (ASSETS / rel).exists():
        return
    write_bytes(ASSETS / rel, src.read_bytes())


def derive(rel, edge=LONG_EDGE, suffix=""):
    with Image.open(ASSETS / rel) as im:
        out = im.convert("RGB")
        out.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        save_image(out, WEB / web_name(rel, suffix), quality=88, optimize=True, progressive=True)


def image_size(rel):
    with Image.open(ASSETS / rel) as im:
        return im.size


def frame_size(rel):
    """Size of the photo as main.py's load() sees it."""
    w, h = image_size(rel)
    s = MAX_SIDE / max(w, h)
    return (round(w * s), round(h * s)) if s < 1 else (w, h)


def points(name):
    return json.loads((ASSETS / "data" / f"{name}.json").read_text())


def poster_rectify(name):
    """main.py's rectify() for one poster, recomputed from the clicked corners:
    the homography, the target rectangle's size, and the output's top-left
    corner in target coordinates. Checked against the output's size."""
    (corners,) = points(f"{name}_corners")
    c = np.array(corners)
    text = (ASSETS / "data" / f"{name}_system.txt").read_text()
    # The aspect ratio is set in main.py; read it off the printed target.
    tw, th = map(float, re.findall(r"-> \(([-\d.]+), ([-\d.]+)\)", text)[2])
    height = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    width = height * tw / th
    rect = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    A, b = [], []
    for (x, y), (xp, yp) in zip(c, rect):
        A += [[x, y, 1, 0, 0, 0, -x * xp, -y * xp], [0, 0, 0, x, y, 1, -x * yp, -y * yp]]
        b += [xp, yp]
    H = np.append(np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0], 1).reshape(3, 3)
    W, Hh = frame_size(f"src/{name}.jpg")
    p = H @ np.array([[0, W - 1, W - 1, 0], [0, 0, Hh - 1, Hh - 1], [1, 1, 1, 1]], float)
    xs, ys = p[0] / p[2], p[1] / p[2]
    m = int(max(width, height))
    x0, y0 = max(int(np.floor(xs.min())), -m), max(int(np.floor(ys.min())), -m)
    x1, y1 = min(int(np.ceil(xs.max())), int(width) + m), min(int(np.ceil(ys.max())), int(height) + m)
    if image_size(f"out/a3/{name}_nn.jpg") != (x1 - x0 + 1, y1 - y0 + 1):
        raise SystemExit(f"{name}: output size no longer matches rectify(); update poster_rectify()")
    return H, width, height, x0, y0


def poster_geometry(name):
    """Where rectify() put the clicked rectangle in its output, as (left, top,
    width, height) in output pixels."""
    _, width, height, x0, y0 = poster_rectify(name)
    return -x0, -y0, width, height


def rect_box(name):
    left, top, w, h = poster_geometry(name)
    return left, top, left + round(w), top + round(h)


def detail_box(name):
    left, top, w, h = poster_geometry(name)
    u, v, frac = DETAIL[name]
    cx, cy, cw = left + u * w, top + v * h, frac * w
    return tuple(round(t) for t in (cx - cw / 2, cy - cw / 4, cx + cw / 2, cy + cw / 4))


def save_detail(crop, rel):
    (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
    save_image(crop.resize((crop.width * DETAIL_SCALE, crop.height * DETAIL_SCALE), Image.Resampling.NEAREST), ASSETS / rel)
    # The page enlarges this with image-rendering: pixelated, so the web copy
    # keeps one pixel per output pixel, losslessly.
    save_image(crop, WEB / web_name(rel).replace(".jpg", ".png"))


def derive_rect(name):
    """The rectified output cropped to the clicked rectangle."""
    for kind in ("nn", "bilinear"):
        with Image.open(ASSETS / "out" / "a3" / f"{name}_{kind}.jpg") as im:
            crop = im.convert("RGB").crop(rect_box(name))
        rel = f"derived/{name}_{kind}_rect.jpg"
        (ASSETS / rel).parent.mkdir(parents=True, exist_ok=True)
        save_image(crop, ASSETS / rel, quality=95)
        derive(rel)


def derive_detail(name):
    for kind in ("nn", "bilinear"):
        with Image.open(ASSETS / "out" / "a3" / f"{name}_{kind}.jpg") as im:
            crop = im.convert("RGB").crop(detail_box(name))
        save_detail(crop, f"derived/{name}_{kind}_detail.png")


def derive_forward(name):
    """Forward warping, to set beside rectify()'s inverse warping: every source
    pixel is sent through H to its nearest output pixel, and output pixels that
    no source pixel lands on stay black. Same H and output frame as rectify()."""
    H, _, _, x0, y0 = poster_rectify(name)
    W, Hh = frame_size(f"src/{name}.jpg")
    with Image.open(ASSETS / "src" / f"{name}.jpg") as im:
        # As main.py's load(); Lanczos stands in for skimage's anti-aliased rescale.
        src = np.asarray(im.convert("RGB").resize((W, Hh), Image.Resampling.LANCZOS))
    yy, xx = np.mgrid[0:Hh, 0:W]
    p = H @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    u = np.round(p[0] / p[2]).astype(int) - x0
    v = np.round(p[1] / p[2]).astype(int) - y0
    out_w, out_h = image_size(f"out/a3/{name}_nn.jpg")
    hit = (p[2] > 0) & (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    out = np.zeros((out_h, out_w, 3), np.uint8)
    out[v[hit], u[hit]] = src.reshape(-1, 3)[hit]
    warped = Image.fromarray(out)
    rel = f"derived/{name}_forward.png"
    save_image(warped, ASSETS / rel, optimize=True)
    derive(rel)
    rel = f"derived/{name}_forward_rect.png"
    save_image(warped.crop(rect_box(name)), ASSETS / rel, optimize=True)
    derive(rel)
    save_detail(warped.crop(detail_box(name)), f"derived/{name}_forward_detail.png")


# ---- generated figures -----------------------------------------------------

def pct(v, total):
    return f"{v / total * 100:.3f}%"


def dark_ink(hex_colour):
    """Whether a number on this fill reads better in dark ink than in white."""
    def lum(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lum(hex_colour[i:i + 2]) for i in (1, 3, 5))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (y + 0.05) / 0.0603 > 1.05 / (y + 0.05)  # vs #1a1a1a (luminance 0.0103)


def zoom_rect(pts, frame):
    """A window around the points with the photo's own aspect ratio."""
    W, H = frame
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 0.08 * max(max(xs) - min(xs), max(ys) - min(ys)) + 0.02 * max(W, H)
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    w = min(max(x1 - x0, (y1 - y0) * W / H), W)
    h = w * H / W
    x = min(max((x0 + x1 - w) / 2, 0), W - w)
    y = min(max((y0 + y1 - h) / 2, 0), H - h)
    return x, y, w, h


def corr_view(rel, pts, frame, zoom=None, quad=False):
    W, H = frame
    style = ""
    if zoom:
        x, y, w, _ = zoom
        style = f' style="--ar:{W} / {H}; --zx:{pct(x, W)}; --zy:{pct(y, H)}; --zs:{W / w:.4f}"'
    # Zoomable views open zoomed in, so they load a photo sharp enough for it.
    src = web(rel, "--2048") if zoom else web(rel)
    view = "corr-view corr-view--zoom" if zoom else "corr-view"
    lines = [f'<div class="{view}"{style}><div class="corr-stage">',
             f'<img src="{src}" data-fullres="assets/proj04/{rel}" width="{W}" height="{H}" alt="" loading="lazy">']
    if quad:
        poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        lines.append(f'<svg class="corr-quad" viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true"><polygon points="{poly}"/></svg>')
    for i, (px, py) in enumerate(pts, 1):
        colour = PALETTE[(i - 1) % len(PALETTE)]
        ink = " corr-pt--dark" if dark_ink(colour) else ""
        lines.append(f'<span class="corr-pt{ink}" data-i="{i}" style="left:{pct(px, W)}; top:{pct(py, H)}; --c:{colour}">{i}</span>')
    lines.append("</div></div>")
    return "\n".join(lines)


def corr_block(m):
    rels = [f"src/{m}_1.jpg", f"src/{m}_2.jpg"]
    frames = [frame_size(rel) for rel in rels]
    sets = points(f"{m}_1_{m}_2")
    zooms = [zoom_rect(pts, frame) for pts, frame in zip(sets, frames)]
    zoomable = all(frame[0] / z[2] >= MIN_ZOOM for frame, z in zip(frames, zooms))
    if zoomable:
        for rel in rels:
            derive(rel, MAX_SIDE, "--2048")
    labels = ["Image 1", "Image 2 (reference)"]
    out = [f'<figure class="corr{" is-zoomed" if zoomable else ""}">', '<div class="corr-pair">']
    for rel, pts, frame, z, label in zip(rels, sets, frames, zooms, labels):
        out += ['<div class="corr-item">', corr_view(rel, pts, frame, z if zoomable else None),
                f'<span class="cell-label">{label}</span>', '</div>']
    out.append("</div>")
    if zoomable:
        out += ['<div class="corr-zoom stepper-group" role="group" aria-label="View">',
                '<button type="button" class="viewset-btn" data-view="full" aria-pressed="false">Full frame</button>',
                '<button type="button" class="viewset-btn is-active" data-view="points" aria-pressed="true">Points</button>',
                '</div>']
    out.append("</figure>")
    return "\n".join(out)


def corners_block(name, warped=None, label="Warped"):
    """The clicked corners on the photo, and where they land in the warp."""
    (pts,) = points(f"{name}_corners")
    left, top, w, h = poster_geometry(name)
    landed = [(left + x, top + y) for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
    warped = warped or f"out/a3/{name}_bilinear.jpg"
    views = [("Original", f"src/{name}.jpg", pts, frame_size(f"src/{name}.jpg")),
             (label, warped, landed, image_size(warped))]
    out = ['<figure class="corr corr--labels-top">', '<div class="corr-pair">']
    for label, rel, p, frame in views:
        out += ['<div class="corr-item">', f'<span class="cell-label">{label}</span>',
                corr_view(rel, p, frame, quad=True), '</div>']
    out += ["</div>", "</figure>"]
    return "\n".join(out)


def parse_matrix(text, pattern):
    m = re.search(pattern + r"\n(\[\[.*?\]\])", text, re.S)
    return [row.split() for row in re.findall(r"\[([^\[\]]+)\]", m.group(1))]


def num(tok):
    tok = tok[:-1] if tok.endswith(".") else tok
    return "0" if float(tok) == 0 else tok.replace("-", "−")


def bmat(rows, aug=False):
    body = "\n".join("<tr>" + "".join(f"<td>{num(t)}</td>" for t in row) + "</tr>" for row in rows)
    return f'<span class="bmat{" bmat--aug" if aug else ""}"><table>\n{body}\n</table></span>'


def system_block(name):
    text = (ASSETS / "data" / f"{name}_system.txt").read_text()
    ab = parse_matrix(text, r"\[A \| b\][^\n]*=")
    h = parse_matrix(text, r"\nH =")
    return "\n".join([
        '<div class="hsys">',
        '<div class="hsys-row">',
        '<span class="hsys-name"><math class="inline-math"><mrow><mo>[</mo><mi>A</mi><mo>|</mo><mi>b</mi><mo>]</mo><mo>=</mo></mrow></math></span>',
        f'<div class="bmat-scroll" tabindex="0" role="region" aria-label="[A | b], {len(ab)} by 9">{bmat(ab, aug=True)}</div>',
        '</div>',
        '<div class="hsys-row">',
        '<span class="hsys-name"><math class="inline-math"><mrow><mi>H</mi><mo>=</mo></mrow></math></span>',
        f'<div class="bmat-scroll">{bmat(h)}</div>',
        '</div>',
        '</div>',
    ])


def stats_block(name):
    text = (ASSETS / "data" / f"{name}_system.txt").read_text()
    n = re.match(r"(\d+) correspondences", text).group(1)
    err = re.search(r"reprojection error ([\d.]+)px", text).group(1)
    return (f'<p class="exif params"><span>Correspondences <b>{n}</b></span>'
            f'<span>Mean reprojection error <b>{err} px</b></span></p>')


def timing_block(name):
    text = (ASSETS / "data" / f"{name}_timing.txt").read_text()
    w, h = re.search(r"output (\d+) x (\d+)", text).groups()
    nn = re.search(r"nearest neighbor ([\d.]+)s", text).group(1)
    bil = re.search(r"bilinear ([\d.]+)s", text).group(1)
    return (f'<p class="exif params"><span>Warped output <b>{w} × {h} px</b></span>'
            f'<span>Nearest neighbor <b>{nn} s</b></span><span>Bilinear <b>{bil} s</b></span></p>')


def fill(page, key, body):
    pat = re.compile(rf"(<!-- gen:{re.escape(key)} -->\n).*?(<!-- /gen:{re.escape(key)} -->)", re.S)
    page, n = pat.subn(lambda m: m.group(1) + body + "\n" + m.group(2), page)
    if n != 1:
        raise SystemExit(f"{PAGE.name}: expected one gen:{key} block, found {n}")
    return page


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    photos = [f"{m}_{i}" for m in MOSAICS for i in (1, 2)] + POSTERS
    for name in photos:
        copy(BASE / "assets" / f"{name}.jpg", f"src/{name}.jpg")
        derive(f"src/{name}.jpg")
    for p in sorted((BASE / "data").glob("*.json")):
        copy(p, f"data/{p.name}")
    for p in sorted((BASE / "out").glob("a[23]/*.txt")):
        copy(p, f"data/{p.name}")

    for name in POSTERS:
        for kind in ("nn", "bilinear"):
            copy(BASE / "out" / "a3" / f"{name}_{kind}.jpg", f"out/a3/{name}_{kind}.jpg")
            derive(f"out/a3/{name}_{kind}.jpg")
        derive_rect(name)
        derive_detail(name)
    for name in FORWARD:
        derive_forward(name)
    for m in MOSAICS:
        for i in (1, 2):
            for kind in ("warped", "weight"):
                rel = f"out/a4/{m}_{m}_{i}_{kind}.jpg"
                copy(BASE / rel, rel)
                derive(rel)
        for kind in ("feathered", "two_band"):
            rel = f"out/a4/{m}_{kind}.jpg"
            copy(BASE / rel, rel)
            derive(rel, RESULT_LONG_EDGE)

    page = PAGE.read_text()
    for m in MOSAICS:
        page = fill(page, f"corr:{m}", corr_block(m))
        page = fill(page, f"stats:{m}", stats_block(f"{m}_1_{m}_2"))
        page = fill(page, f"system:{m}", system_block(f"{m}_1_{m}_2"))
    for name in POSTERS:
        page = fill(page, f"corners:{name}", corners_block(name))
        page = fill(page, f"timing:{name}", timing_block(name))
    for name in FORWARD:
        page = fill(page, f"forward:{name}", corners_block(name, f"derived/{name}_forward.png", "Forward warped"))
    write_bytes(PAGE, page.encode())


if __name__ == "__main__":
    main()
