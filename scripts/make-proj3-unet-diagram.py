"""Build the Part B UNet diagrams (inline SVG + HTML legend) and splice them into proj3.html.

1.1: the unconditional denoiser (D = 128). 2.1: the time-conditioned UNet (D = 64),
which multiplies the Unflatten and first UpBlock outputs by FCBlock embeddings of t.
Tensors are blocks: the front face is the spatial size, depth the channel count.
Each diagram is written between <!-- {prefix}-diagram:start/end --> markers.
"""
import html
import math
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "proj3.html"
WEB = "assets/proj03/web/"

FACE = {28: 66, 14: 38, 7: 21, 1: 7}            # front-face side, px
CY = {28: 132, 14: 248, 7: 330, 1: 392}         # front-face vertical centre per level
DX, DY = 0.72, -0.5                              # screen offset per px of depth
GAP = 26
X0 = 34
COND = "#e5484d"                                 # time conditioning
CLS = "#b4359c"                                  # class conditioning
EMB = "#c43c6c"                                  # joint (t, c) embedding

OPS = {
    "conv": ("#488af3", "ConvB"),
    "down": ("#2a9d8f", "DownB"),
    "up": ("#8b6cd9", "UpB"),
    "flat": ("#d08a2c", "Flatten"),
    "unflat": ("#c0612b", "Unflatten"),
    "out": ("#333333", "Conv 3×3"),
    "res": ("#0e7c86", "ResB"),
    "attn": ("#c77d12", "Attn"),
    "cat": ("#6b7178", "⊕ skip"),
}
FILL = {"t": ("url(#{p}-g-front)", "url(#{p}-g-top)", "url(#{p}-g-side)", "#5f8de0"),
        "skip": ("url(#{p}-g-sfront)", "#eef0f2", "url(#{p}-g-sside)", "#959ba1"),
        "img": ("#111", "#3a3a3a", "#222", "#111"),
        "plain": ("#3b4046", "#5a6067", "#2b2f34", "#22262a"),
        "attn": ("#fdf0d9", "#fff7ea", "#f0d29a", "#d08a2c")}
EDGES = [("in", "s1"), ("s1", "s2"), ("s2", "s3"), ("s3", "fl"),
         ("fl", "c7"), ("c7", "c14"), ("c14", "c28"), ("c28", "cb"), ("cb", "out")]
SKIPS = [("s1", "c28"), ("s2", "c14"), ("s3", "c7")]


def depth(c):
    return 3 if c == 1 else 3 + c / 8            # 64 -> 11, 128 -> 19, 256 -> 35


def sym(text):
    return f'<tspan class="unet-sym">{text}</tspan>'


def tensors(D, io):
    """io: dict with in/out kinds, images and captions."""
    return [
        ("in", 28, [(1, io["in_kind"])], "1×28×28", "Input", None, io["in_cap"], io["in_tip"], io.get("in_img")),
        ("s1", 28, [(D, "t")], f"{D}×28×28", "ConvB", "conv", None, f"ConvBlock output (skip 1), {D}×28×28", None),
        ("s2", 14, [(D, "t")], f"{D}×14×14", "DownB", "down", None, f"DownBlock output (skip 2), {D}×14×14", None),
        ("s3", 7, [(2 * D, "t")], f"{2 * D}×7×7", "DownB", "down", None, f"DownBlock output (skip 3), {2 * D}×7×7", None),
        ("fl", 1, [(2 * D, "t")], f"{2 * D}×1×1", "Flatten", "flat", None, f"Flatten (average pool) output, {2 * D}×1×1", None),
        ("c7", 7, [(2 * D, "t"), (2 * D, "skip")], f"({2 * D} + {2 * D})×7×7", "Unflatten ⊕ skip", "unflat", None,
         f"Unflatten output {2 * D}×7×7{io['mod7']}, concatenated with skip 3 → {4 * D}×7×7", None),
        ("c14", 14, [(D, "t"), (D, "skip")], f"({D} + {D})×14×14", "UpB ⊕ skip", "up", None,
         f"UpBlock output {D}×14×14{io['mod14']}, concatenated with skip 2 → {2 * D}×14×14", None),
        ("c28", 28, [(D, "t"), (D, "skip")], f"({D} + {D})×28×28", "UpB ⊕ skip", "up", None,
         f"UpBlock output {D}×28×28, concatenated with skip 1 → {2 * D}×28×28", None),
        ("cb", 28, [(D, "t")], f"{D}×28×28", "ConvB", "conv", None, f"ConvBlock output, {D}×28×28", None),
        ("out", 28, [(1, io["out_kind"])], "1×28×28", "Conv 3×3", "out", io["out_cap"], io["out_tip"], io.get("out_img")),
    ]


def poly(pts):
    return " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)


def cuboid(p, x, y, f, d, kind, z0=0, stroke_override=None):
    front, top, side, stroke = (c.format(p=p) for c in FILL[kind])
    stroke = stroke_override or stroke
    fx, fy = x + z0 * DX, y + z0 * DY
    ex, ey = d * DX, d * DY
    return [
        f'<polygon points="{poly([(fx, fy), (fx + ex, fy + ey), (fx + f + ex, fy + ey), (fx + f, fy)])}" fill="{top}" stroke="{stroke}"/>',
        f'<polygon points="{poly([(fx + f, fy), (fx + f + ex, fy + ey), (fx + f + ex, fy + f + ey), (fx + f, fy + f)])}" fill="{side}" stroke="{stroke}"/>',
        f'<rect x="{fx:.1f}" y="{fy:.1f}" width="{f:.1f}" height="{f:.1f}" fill="{front}" stroke="{stroke}"/>',
    ]


def chevron(mid, color):
    # open chevron, fixed size in user units, tip exactly at the path's end
    return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" '
            'markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
            f'<path d="M3.5 1.5 L9 5 L3.5 8.5" fill="none" stroke="{color}" stroke-width="1.5" '
            'stroke-linecap="round" stroke-linejoin="round"/></marker>')


def defs(p):
    return (chevron(f"{p}-ah", "#6b7178") + chevron(f"{p}-ah-skip", "#9aa0a6") + chevron(f"{p}-ah-cond", COND) + chevron(f"{p}-ah-cls", CLS) + chevron(f"{p}-ah-emb", EMB)
            + f'<linearGradient id="{p}-g-front" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e6effe"/><stop offset="1" stop-color="#c3d6f8"/></linearGradient>'
            f'<linearGradient id="{p}-g-top" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#f4f8ff"/><stop offset="1" stop-color="#dde8fc"/></linearGradient>'
            f'<linearGradient id="{p}-g-side" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#a9c2f1"/><stop offset="1" stop-color="#86a8ea"/></linearGradient>'
            f'<linearGradient id="{p}-g-sfront" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#eef0f2"/><stop offset="1" stop-color="#d9dde1"/></linearGradient>'
            f'<linearGradient id="{p}-g-sside" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#c9ced3"/><stop offset="1" stop-color="#b3b9bf"/></linearGradient>'
            f'<filter id="{p}-shadow" x="-20%" y="-20%" width="140%" height="160%"><feDropShadow dx="2" dy="5" stdDeviation="4" flood-color="#1a2b4a" flood-opacity="0.16"/></filter>')


def build(cfg):
    p = cfg["prefix"]
    T = cfg.get("tensors") or tensors(cfg["D"], cfg["io"])
    face, cy = cfg.get("face", FACE), cfg.get("cy", CY)
    skips, edges = cfg.get("skips", SKIPS), cfg.get("edges", EDGES)
    top_level = cfg.get("top_level", 28)
    extra = cfg.get("extra_gap", {})
    label_dx = cfg.get("label_dx", {"c7": 14, "c14": 10})
    geo, x = {}, X0
    for name, lvl, slabs, *_ in T:
        x += extra.get(name, 0)
        f = face[lvl]
        d = sum(depth(c) for c, _ in slabs)
        geo[name] = dict(x=x, y=cy[lvl] - f / 2, f=f, d=d, lvl=lvl, slabs=slabs)
        x += f + d * DX + GAP
    W = math.ceil(x - GAP + X0)
    parts = []

    for a, b in skips:
        A, B = geo[a], geo[b]
        ax = A["x"] + A["f"] / 2 + A["d"] / 2 * DX
        ay = A["y"] + A["d"] / 2 * DY
        first = depth(B["slabs"][0][0])
        mid = first + (B["d"] - first) / 2
        bx, by = B["x"] + B["f"] / 2 + mid * DX, B["y"] + mid * DY
        top = min(ay, by) - (26 if A["lvl"] == top_level else 18)
        parts.append(f'<path class="unet-skip" d="M{ax:.1f},{ay:.1f} V{top:.1f} H{bx:.1f} V{by - 3:.1f}" marker-end="url(#{p}-ah-skip)"/>')

    for name, lvl, slabs, dims, op, opk, cap, tip, img in T:
        g = geo[name]
        parts.append(f'<g class="unet-t" filter="url(#{p}-shadow)"><title>{html.escape(tip)}</title>')
        zs, z = [], 0
        for c, kind in slabs:
            zs.append((z, depth(c), kind)); z += depth(c)
        scaled = name in cfg.get("modulate", {}) or name in cfg.get("cond_rows", {})
        outline = cfg.get("outline", COND)
        for k, (z0, d, kind) in reversed(list(enumerate(zs))):
            parts += cuboid(p, g["x"], g["y"], g["f"], d, kind, z0, outline if scaled and k == 0 else None)
        if img:
            parts.append(f'<image href="{WEB}{img}" x="{g["x"]:.1f}" y="{g["y"]:.1f}" width="{g["f"]}" height="{g["f"]}" preserveAspectRatio="none" class="unet-img"/>')
            parts.append(f'<rect x="{g["x"]:.1f}" y="{g["y"]:.1f}" width="{g["f"]}" height="{g["f"]}" fill="none" stroke="#111"/>')
        if name in cfg.get("emb_blocks", ()):
            parts.append(f'<circle class="unet-emb-badge" cx="{g["x"] + g["f"]:.1f}" cy="{g["y"]:.1f}" r="4.5"><title>Receives the (t, c) embedding</title></circle>')
        parts.append("</g>")
        cx = g["x"] + g["f"] / 2 + g["d"] * DX / 2 + label_dx.get(name, 0)
        oplines = list(op) if isinstance(op, (list, tuple)) else [(line, opk) for line in op.split("\n")]
        ly0 = g["y"] + g["f"] + 18
        for k, (line, key) in enumerate(oplines):
            col = OPS[key][0] if key else "#333333"
            parts.append(f'<text class="unet-opname" x="{cx:.1f}" y="{ly0 + k * 14:.1f}" text-anchor="middle" fill="{col}">{html.escape(line)}</text>')
        ly = ly0 + (len(oplines) - 1) * 14
        if scaled:
            dims = re.sub(r"^\((\d+)", lambda m: f'(<tspan class="unet-scaled" style="fill:{outline}">{m.group(1)}</tspan>', dims)
        parts.append(f'<text class="unet-dim" x="{cx:.1f}" y="{ly + 15:.1f}" text-anchor="middle">{dims}</text>')
        if cap:
            parts.append(f'<text class="unet-io" x="{cx:.1f}" y="{ly + 30:.1f}" text-anchor="middle">{cap}</text>')

    mods = cfg.get("modulate", {})          # target tensor -> FCB output size
    rows = cfg.get("cond_rows", {})         # target tensor -> [(op, input, size)], left to right
    mod_pts, node_pts = {}, {}
    for a, b in edges:
        A, B = geo[a], geo[b]
        sx = A["x"] + A["f"] + A["d"] * DX + 3
        ex = B["x"] - 3
        sy = A["y"] + A["f"] / 2
        ey = B["y"] + B["f"] / 2
        if abs(sy - ey) < 1:
            d = f"M{sx:.1f},{sy:.1f} H{ex:.1f}"
        else:
            # turn early when a multiply node sits on the last leg, so it has room
            early = b in mods or b in rows
            mx = sx + 9 if early else (sx + ex) / 2
            r = min(7, abs(ey - sy) / 2, (mx - sx) / 1.5 if early else (ex - sx) / 4)
            sg = 1 if ey > sy else -1
            d = (f"M{sx:.1f},{sy:.1f} H{mx - r:.1f} Q{mx:.1f},{sy:.1f} {mx:.1f},{sy + sg * r:.1f} "
                 f"V{ey - sg * r:.1f} Q{mx:.1f},{ey:.1f} {mx + r:.1f},{ey:.1f} H{ex:.1f}")
            if b in mods:
                mod_pts[b] = ((mx + ex) / 2 + 1, ey)
            if b in rows:
                node_pts[b] = [(mx + 18 + k * 24, ey, op) for k, op in enumerate(rows[b])]
        parts.append(f'<path class="unet-op" d="{d}" marker-end="url(#{p}-ah)"/>')

    bottom = max(g["y"] + g["f"] for g in geo.values()) + 46
    if mods:
        # The conditioning sits in the empty space right of the decoder. Each FCB
        # sends its output left along its own lane under the decoder, then up into
        # the multiply node on that decoder edge.
        bw, bh, r = 74, 34, 6
        lanes = {tgt: geo[tgt]["y"] + geo[tgt]["f"] + 18 + 15 + 16 for tgt in mods}
        c14 = geo["c14"]
        left = c14["x"] + c14["f"] + c14["d"] * DX + 58
        fx = left + 14 + bw / 2                       # FCB centres share one column
        tx = fx + bw / 2 + 64
        ty = sum(lanes.values()) / len(lanes)
        right = tx + 13 + 18
        top = min(lanes.values()) - bh / 2 - 30
        bot = max(lanes.values()) + bh / 2 + 16
        W = max(W, math.ceil(right + 8))
        cond = [f'<rect class="unet-cond-box" x="{left:.1f}" y="{top:.1f}" width="{right - left:.1f}" height="{bot - top:.1f}" rx="10"/>',
                f'<text class="unet-cond-title" x="{left + 12:.1f}" y="{top + 17:.1f}">Time conditioning</text>']
        # t feeds both FCBs through a short vertical bus
        busx = fx + bw / 2 + 24
        cond.append(f'<path class="unet-cond-line" d="M{tx - 13:.1f},{ty:.1f} H{busx:.1f}"/>')
        cond.append(f'<path class="unet-cond-line" d="M{busx:.1f},{min(lanes.values()):.1f} V{max(lanes.values()):.1f}"/>')
        for tgt, lane in lanes.items():
            cond.append(f'<path class="unet-cond-line" d="M{busx:.1f},{lane:.1f} H{fx + bw / 2 + 3:.1f}" marker-end="url(#{p}-ah-cond)"/>')
        cond.append(f'<circle class="unet-cond-node" cx="{tx:.1f}" cy="{ty:.1f}" r="13"/>')
        cond.append(f'<text class="unet-cond-t" x="{tx:.1f}" y="{ty + 5:.1f}" text-anchor="middle">{sym("t")}</text>')
        for tgt, size in mods.items():
            lane = lanes[tgt]
            mxp, myp = mod_pts[tgt]
            cond.append(f'<path class="unet-cond-line" d="M{fx - bw / 2:.1f},{lane:.1f} H{mxp + r:.1f} '
                        f'Q{mxp:.1f},{lane:.1f} {mxp:.1f},{lane - r:.1f} V{myp + 11:.1f}" marker-end="url(#{p}-ah-cond)"/>')
            cond.append(f'<rect class="unet-cond-fcb" x="{fx - bw / 2:.1f}" y="{lane - bh / 2:.1f}" width="{bw}" height="{bh}" rx="7"/>')
            cond.append(f'<text class="unet-cond-name" x="{fx:.1f}" y="{lane - 2:.1f}" text-anchor="middle">FCB</text>')
            cond.append(f'<text class="unet-cond-dim" x="{fx:.1f}" y="{lane + 11:.1f}" text-anchor="middle">1 → {size}</text>')
            cond.append(f'<g class="unet-mul"><title>Multiply channel-wise by FCB(t), {size} channels</title>'
                        f'<circle cx="{mxp:.1f}" cy="{myp:.1f}" r="8"/>'
                        f'<path d="M{mxp - 3.6:.1f},{myp - 3.6:.1f} L{mxp + 3.6:.1f},{myp + 3.6:.1f} M{mxp + 3.6:.1f},{myp - 3.6:.1f} L{mxp - 3.6:.1f},{myp + 3.6:.1f}"/></g>')
        parts += cond
        bottom = max(bottom, bot + 8)

    if rows:
        # One row per FCB, each with its own input on the right, so no lines cross.
        # Per decoder edge the rightmost node takes the highest lane.
        bw, bh, r = 104, 24, 6
        lanes = []
        for tgt, nodes in node_pts.items():
            g = geo[tgt]
            base = g["y"] + g["f"] + 18 + 15 + 16
            for k, (nx, ny, (op, src, size)) in enumerate(nodes):
                lanes.append((base + (len(nodes) - 1 - k) * 30, nx, ny, op, src, size))
        c14 = geo["c14"]
        left = c14["x"] + c14["f"] + c14["d"] * DX + 58
        fx = left + 14 + bw / 2
        ix = fx + bw / 2 + 36
        right = ix + 11 + 16
        top = min(l[0] for l in lanes) - bh / 2 - 30
        bot = max(l[0] for l in lanes) + bh / 2 + 14
        W = max(W, math.ceil(right + 8))
        cond = [f'<rect class="unet-cond-box unet-cond-box--neutral" x="{left:.1f}" y="{top:.1f}" width="{right - left:.1f}" height="{bot - top:.1f}" rx="10"/>',
                f'<text class="unet-cond-title" x="{left + 12:.1f}" y="{top + 17:.1f}"><tspan fill="{CLS}">Class</tspan>'
                f'<tspan fill="#6b7178"> and </tspan><tspan fill="{COND}">time</tspan><tspan fill="#6b7178"> conditioning</tspan></text>']
        for lane, nx, ny, op, src, size in lanes:
            col, mk = (CLS, "cls") if src == "c" else (COND, "cond")
            fin = "10" if src == "c" else "1"
            cond.append(f'<path class="unet-cond-line" stroke="{col}" d="M{fx - bw / 2:.1f},{lane:.1f} H{nx + r:.1f} '
                        f'Q{nx:.1f},{lane:.1f} {nx:.1f},{lane - r:.1f} V{ny + 11:.1f}" marker-end="url(#{p}-ah-{mk})"/>')
            cond.append(f'<path class="unet-cond-line" stroke="{col}" d="M{ix - 11:.1f},{lane:.1f} H{fx + bw / 2 + 3:.1f}" marker-end="url(#{p}-ah-{mk})"/>')
            cond.append(f'<rect class="unet-cond-fcb" style="stroke:{col}; fill:{"#fdf3fa" if src == "c" else "#fff5f5"}" x="{fx - bw / 2:.1f}" y="{lane - bh / 2:.1f}" width="{bw}" height="{bh}" rx="6"/>')
            cond.append(f'<text class="unet-cond-row" x="{fx:.1f}" y="{lane + 4:.1f}" text-anchor="middle" fill="{col}">'
                        f'<tspan font-weight="700">FCB</tspan><tspan class="unet-sub" dy="3" fill="{col}">{src}</tspan>'
                        f'<tspan dy="-3"> {fin} → {size}</tspan></text>')
            cond.append(f'<circle class="unet-cond-node" style="stroke:{col}" cx="{ix:.1f}" cy="{lane:.1f}" r="11"/>')
            cond.append(f'<text class="unet-cond-t" x="{ix:.1f}" y="{lane + 4.5:.1f}" text-anchor="middle"><tspan class="unet-sym" fill="{col}">{src}</tspan></text>')
            glyph = (f'M{nx - 3.6:.1f},{ny - 3.6:.1f} L{nx + 3.6:.1f},{ny + 3.6:.1f} M{nx + 3.6:.1f},{ny - 3.6:.1f} L{nx - 3.6:.1f},{ny + 3.6:.1f}'
                     if op == "mul" else f'M{nx - 4.4:.1f},{ny:.1f} H{nx + 4.4:.1f} M{nx:.1f},{ny - 4.4:.1f} V{ny + 4.4:.1f}')
            what = f"Multiply channel-wise by FCB_c(c), {size} channels" if op == "mul" else f"Add FCB_t(t) to each channel, {size} channels"
            cond.append(f'<g class="unet-mul" style="--node:{col}"><title>{what}</title><circle cx="{nx:.1f}" cy="{ny:.1f}" r="8" style="stroke:{col}"/>'
                        f'<path d="{glyph}" style="stroke:{col}"/></g>')
        parts += cond
        bottom = max(bottom, bot + 8)

    if cfg.get("emb_box"):
        # t and c each through an FCB to 512, summed, GELU: one embedding for every ResB
        anchor = geo[cfg["emb_box"]]
        bl = anchor["x"] + anchor["f"] + anchor["d"] * DX + 46
        y1 = cfg.get("emb_y", 300); y2 = y1 + 44; ym = (y1 + y2) / 2
        bw, bh = 96, 24
        cxn, fx0 = bl + 24, bl + 46
        sx_ = fx0 + bw + 26                       # sum node
        gx0, gw = sx_ + 22, 48                    # GELU pill
        ex0, ew = gx0 + gw + 18, 78               # embedding pill
        br = ex0 + ew + 14
        top, bot = y1 - bh / 2 - 30, y2 + bh / 2 + 34
        W = max(W, math.ceil(br + 8))
        e = [f'<rect class="unet-cond-box unet-cond-box--neutral" x="{bl:.1f}" y="{top:.1f}" width="{br - bl:.1f}" height="{bot - top:.1f}" rx="10"/>',
             f'<text class="unet-cond-title" x="{bl + 12:.1f}" y="{top + 17:.1f}"><tspan fill="{CLS}">Class</tspan><tspan fill="#6b7178"> and </tspan>'
             f'<tspan fill="{COND}">time</tspan><tspan fill="#6b7178"> embedding</tspan></text>']
        for yy, src, col, mk, fin in ((y1, "t", COND, "cond", "1"), (y2, "c", CLS, "cls", "10")):
            e.append(f'<path class="unet-cond-line" stroke="{col}" d="M{cxn + 11:.1f},{yy:.1f} H{fx0 - 3:.1f}" marker-end="url(#{p}-ah-{mk})"/>')
            e.append(f'<circle class="unet-cond-node" style="stroke:{col}" cx="{cxn:.1f}" cy="{yy:.1f}" r="11"/>')
            e.append(f'<text class="unet-cond-t" x="{cxn:.1f}" y="{yy + 4.5:.1f}" text-anchor="middle"><tspan class="unet-sym" fill="{col}">{src}</tspan></text>')
            e.append(f'<rect class="unet-cond-fcb" style="stroke:{col}; fill:{"#fdf3fa" if src == "c" else "#fff5f5"}" x="{fx0:.1f}" y="{yy - bh / 2:.1f}" width="{bw}" height="{bh}" rx="6"/>')
            e.append(f'<text class="unet-cond-row" x="{fx0 + bw / 2:.1f}" y="{yy + 4:.1f}" text-anchor="middle" fill="{col}"><tspan font-weight="700">FCB</tspan>'
                     f'<tspan class="unet-sub" dy="3" fill="{col}">{src}</tspan><tspan dy="-3"> {fin} → 512</tspan></text>')
            # into the sum node from above / below
            sg = 1 if yy < ym else -1
            e.append(f'<path class="unet-cond-line" stroke="{col}" d="M{fx0 + bw:.1f},{yy:.1f} H{sx_ - 6:.1f} Q{sx_:.1f},{yy:.1f} {sx_:.1f},{yy + sg * 6:.1f} '
                     f'V{ym - sg * 11:.1f}" marker-end="url(#{p}-ah-{mk})"/>')
        e.append(f'<g class="unet-mul"><circle cx="{sx_:.1f}" cy="{ym:.1f}" r="8" style="stroke:{EMB}"/>'
                 f'<path d="M{sx_ - 4.4:.1f},{ym:.1f} H{sx_ + 4.4:.1f} M{sx_:.1f},{ym - 4.4:.1f} V{ym + 4.4:.1f}" style="stroke:{EMB}"/></g>')
        e.append(f'<path class="unet-cond-line" stroke="{EMB}" d="M{sx_ + 8:.1f},{ym:.1f} H{gx0 - 3:.1f}" marker-end="url(#{p}-ah-emb)"/>')
        e.append(f'<rect class="unet-emb-pill unet-emb-pill--gelu" x="{gx0:.1f}" y="{ym - 11:.1f}" width="{gw}" height="22" rx="11"/>')
        e.append(f'<text class="unet-emb-text" x="{gx0 + gw / 2:.1f}" y="{ym + 4:.1f}" text-anchor="middle" fill="{EMB}">GELU</text>')
        e.append(f'<path class="unet-cond-line" stroke="{EMB}" d="M{gx0 + gw:.1f},{ym:.1f} H{ex0 - 3:.1f}" marker-end="url(#{p}-ah-emb)"/>')
        e.append(f'<rect class="unet-emb-pill" x="{ex0:.1f}" y="{ym - 12:.1f}" width="{ew}" height="24" rx="12"/>')
        e.append(f'<text class="unet-emb-text unet-emb-text--on" x="{ex0 + ew / 2:.1f}" y="{ym + 4.5:.1f}" text-anchor="middle">emb · 512</text>')
        e.append(f'<circle class="unet-emb-badge" cx="{bl + 18:.1f}" cy="{bot - 15:.1f}" r="4.5"/>')
        e.append(f'<text class="unet-emb-note" x="{bl + 28:.1f}" y="{bot - 11:.1f}">scale &amp; shift inside every ResBlock of a marked block</text>')
        parts += e
        bottom = max(bottom, bot + 8)

    ys = [g["y"] + min(0.0, g["d"] * DY) for g in geo.values()]
    top_y = min(ys) - 40
    H = bottom - top_y

    svg = (f'<svg class="unet-svg" viewBox="0 {top_y:.0f} {W} {H:.0f}" role="img" aria-labelledby="{p}-title {p}-desc">\n'
           f'<title id="{p}-title">{html.escape(cfg["title"])}</title>\n'
           f'<desc id="{p}-desc">{html.escape(cfg["desc"])}</desc>\n'
           f'<defs>{defs(p)}</defs>\n' + "\n".join(parts) + "\n</svg>")

    legend = ['<dl class="unet-legend">']
    for k, desc in cfg["legend"]:
        if isinstance(k, tuple) and k[0] in OPS:          # an op shown under another name
            c, name = OPS[k[0]][0], k[1]
            legend.append(f'<div><dt><span class="unet-swatch" style="--c:{c}"></span>{name}</dt><dd>{desc}</dd></div>')
        elif k in OPS:
            c, name = OPS[k]
            legend.append(f'<div><dt><span class="unet-swatch" style="--c:{c}"></span>{name}</dt><dd>{desc}</dd></div>')
        else:
            swatch, name = k
            legend.append(f'<div><dt>{swatch}{name}</dt><dd>{desc}</dd></div>')
    legend.append('<div><dt><span class="unet-swatch unet-swatch--skip"></span>Skip</dt><dd>Copy the encoder tensor and concatenate along channels (grey slab)</dd></div>')
    legend.append('</dl>')

    return (f'<figure class="unet-fig">\n<div class="unet-scroll" tabindex="0" role="region" aria-label="{html.escape(cfg["title"])}">\n'
            + svg + '\n</div>\n' + "\n".join(legend) + f'\n<p class="table-note">{cfg["note"]}</p>\n</figure>')


BASE_LEGEND = [
    ("conv", "ConvBlock: (Conv 3×3 → BatchNorm → GELU) × 2"),
    ("down", "DownBlock: Conv 3×3, stride 2 → BatchNorm → GELU → ConvB"),
    ("up", "UpBlock: ConvTranspose 4×4, stride 2 → BatchNorm → GELU → ConvB"),
    ("flat", "AvgPool 7×7 → GELU"),
    ("unflat", "ConvTranspose 7×7, stride 7 → BatchNorm → GELU"),
]

UNCOND = dict(
    prefix="unet", D=128,
    io=dict(in_kind="img", in_img="b--derived--unet_in.jpg", in_cap=f"noisy {sym('z')}", in_tip="Noisy input z, 1×28×28",
            out_kind="img", out_img="b--derived--unet_out.jpg", out_cap=f"denoised {sym('x̂')}", out_tip="Denoised output x̂, 1×28×28",
            mod7="", mod14=""),
    title="Unconditional UNet used as the single-step denoiser",
    desc="A noisy 1×28×28 digit passes through ConvBlock to 128×28×28, DownBlock to 128×14×14, DownBlock to 256×7×7, "
         "Flatten to 256×1×1 and Unflatten back to 256×7×7. Each decoder level concatenates the matching encoder tensor, "
         "then UpBlocks return to 14×14 and 28×28, a ConvBlock gives 128×28×28 and a 3×3 convolution outputs the denoised 1×28×28 digit.",
    legend=BASE_LEGEND + [("out", "Conv 3×3, 128 → 1 channel")],
    note="Front face: spatial size. Depth: channels. Hidden width D = 128; no time or class conditioning.",
)

TIME = dict(
    prefix="tunet", D=64,
    io=dict(in_kind="img", in_img="b--derived--tunet_in.jpg", in_cap=f"noisy {sym('x')}<tspan class=\"unet-sub\" dy=\"4\">t</tspan>",
            in_tip="Noisy input x_t = (1 − t) x_0 + t x_1, 1×28×28",
            out_kind="plain", out_cap=f"flow {sym('u')}<tspan class=\"unet-sub\" dy=\"4\">θ</tspan>",
            out_tip="Predicted flow u_θ(x_t, t), 1×28×28, trained against x_1 − x_0",
            mod7=" multiplied by FCB(t)", mod14=" multiplied by FCB(t)"),
    modulate={"c7": 128, "c14": 64},
    extra_gap={"c7": 34, "c14": 54},
    title="Time-conditioned UNet for flow matching",
    desc="The same UNet with hidden width 64: 1×28×28 input x_t through 64×28×28, 64×14×14, 128×7×7 and 128×1×1, then back up. "
         "The scalar time t passes through two FCBlocks, 1 → 128 and 1 → 64; their outputs multiply the Unflatten output (128×7×7) "
         "and the first UpBlock output (64×14×14) channel-wise. The network outputs the predicted flow u_θ(x_t, t), 1×28×28.",
    legend=BASE_LEGEND + [
        ("out", "Conv 3×3, 64 → 1 channel"),
        (('<span class="unet-swatch" style="--c:#e5484d"></span>', "FCB"), "FCBlock: Linear(1 → C) → GELU → Linear(C → C)"),
        (('<span class="unet-mul-swatch" aria-hidden="true">×</span>', "Multiply"), "Scale each channel of the red-outlined slab by the FCB output, broadcast over space, before the skip is concatenated"),
    ],
    note="Front face: spatial size. Depth: channels. Hidden width D = 64. Red: time conditioning, t ∈ [0, 1]. "
         "Input shown: a test digit mixed with noise at t = 0.5, for illustration; the output is a flow, not an image.",
)


CLASS = dict(
    prefix="cunet", D=64,
    io=dict(TIME["io"], mod7=" multiplied by FCB_c(c) plus FCB_t(t)", mod14=" multiplied by FCB_c(c) plus FCB_t(t)"),
    cond_rows={"c7": [("mul", "c", 128), ("add", "t", 128)], "c14": [("mul", "c", 64), ("add", "t", 64)]},
    outline=CLS,
    extra_gap={"c7": 52, "c14": 72},
    title="Class- and time-conditioned UNet for flow matching",
    desc="The time-conditioned UNet with a class input as well. The class c is a one-hot vector of length 10 and t a scalar; four FCBlocks embed them. "
         "At 7×7 the Unflatten output (128×7×7) is multiplied channel-wise by FCB_c(c) (10 → 128) and then FCB_t(t) (1 → 128) is added; "
         "at 14×14 the first UpBlock output (64×14×14) is treated the same way with 10 → 64 and 1 → 64. The output is the predicted flow u_θ(x_t, t, c).",
    legend=BASE_LEGEND + [
        ("out", "Conv 3×3, 64 → 1 channel"),
        ((f'<span class="unet-swatch" style="--c:{CLS}"></span>', "FCB<sub>c</sub>"), "FCBlock on the one-hot class vector: Linear(10 → C) → GELU → Linear(C → C)"),
        ((f'<span class="unet-swatch" style="--c:{COND}"></span>', "FCB<sub>t</sub>"), "FCBlock on t: Linear(1 → C) → GELU → Linear(C → C)"),
        ((f'<span class="unet-mul-swatch" style="border-color:{CLS}; color:{CLS}" aria-hidden="true">×</span>', "Multiply"), "Scale each channel of the outlined slab by FCB<sub>c</sub>(c)"),
        ((f'<span class="unet-mul-swatch" style="border-color:{COND}; color:{COND}" aria-hidden="true">+</span>', "Add"), "Then add FCB<sub>t</sub>(t) to each channel, broadcast over space, before the skip is concatenated"),
    ],
    note="Front face: spatial size. Depth: channels. Hidden width D = 64. c: one-hot class vector, set to zero 10% of the time in training "
         "(p<sub>uncond</sub> = 0.1); t ∈ [0, 1]. Input shown: a test digit mixed with noise at t = 0.5, for illustration.",
)


DEEP = dict(
    prefix="dunet", D=128,
    face={32: 66, 16: 38, 8: 21, "mid": 21},
    cy={32: 132, 16: 252, 8: 336, "mid": 410},
    top_level=32,
    tensors=[
        ("in", 32, [(3, "img")], "3×32×32", "Input", None, f"noisy {sym('x')}<tspan class=\"unet-sub\" dy=\"4\">t</tspan>",
         "Noisy input x_t, 3×32×32", "b--derived--dunet_in.jpg"),
        ("s32", 32, [(128, "t")], "128×32×32", [("Stem conv →", "out"), ("2× ResB", "res")], "res", None, "Stem conv, then two ResBlocks (skip 1), 128×32×32", None),
        ("s16", 16, [(256, "t")], "256×16×16", "Down →\n2× ResB", "down", None, "Stride-2 conv, then two ResBlocks (skip 2), 256×16×16", None),
        ("s8", 8, [(256, "t")], "256×8×8", "Down →\n2× ResB", "down", None, "Stride-2 conv, then two ResBlocks (skip 3), 256×8×8", None),
        ("mid", "mid", [(256, "attn")], "256×8×8", "ResB → Attn → ResB", "attn", None,
         "Bottleneck at 8×8: ResBlock, multi-head self-attention, ResBlock, 256×8×8", None),
        ("c8", 8, [(256, "t"), (256, "skip")], "(256 + 256)×8×8", "⊕ skip", "cat", None, "Bottleneck output concatenated with skip 3 → 512×8×8", None),
        ("c16", 16, [(256, "t"), (256, "skip")], "(256 + 256)×16×16", "2× ResB → Up\n⊕ skip", "up", None,
         "Two ResBlocks (512 → 256 at 8×8), transposed conv to 256×16×16, concatenated with skip 2 → 512×16×16", None),
        ("c32", 32, [(128, "t"), (128, "skip")], "(128 + 128)×32×32", "2× ResB → Up\n⊕ skip", "up", None,
         "Two ResBlocks (512 → 256 at 16×16), transposed conv to 128×32×32, concatenated with skip 1 → 256×32×32", None),
        ("o32", 32, [(128, "t")], "128×32×32", "2× ResB", "res", None, "Two ResBlocks (256 → 128), 128×32×32", None),
        ("out", 32, [(3, "plain")], "3×32×32", "Output conv", "out", f"flow {sym('u')}<tspan class=\"unet-sub\" dy=\"4\">θ</tspan>",
         "Predicted flow u_θ(x_t, t, c), 3×32×32", None),
    ],
    edges=[("in", "s32"), ("s32", "s16"), ("s16", "s8"), ("s8", "mid"), ("mid", "c8"), ("c8", "c16"), ("c16", "c32"), ("c32", "o32"), ("o32", "out")],
    skips=[("s32", "c32"), ("s16", "c16"), ("s8", "c8")],
    emb_blocks=("s32", "s16", "s8", "mid", "c16", "c32", "o32"),
    emb_box="c16", emb_y=300,
    label_dx={"c8": 6},
    extra_gap={"c16": 14, "c32": 18},
    title="Wider and deeper class- and time-conditioned UNet for CIFAR-10",
    desc="A 3×32×32 input x_t goes through a stem conv and two ResBlocks (128×32×32), a stride-2 conv and two ResBlocks (256×16×16), "
         "and another stride-2 conv and two ResBlocks (256×8×8). The 8×8 bottleneck is ResBlock, multi-head self-attention, ResBlock. "
         "The decoder concatenates each skip, applies two ResBlocks and a transposed conv per level, and a final 3×3 conv outputs the 3×32×32 flow. "
         "t and c each pass through an FCBlock to 512 dimensions; their sum, after GELU, sets a per-channel scale and shift in every ResBlock.",
    legend=[
        ("res", "ResBlock: Conv 3×3 → BN → GELU → Conv 3×3 → BN, scaled and shifted by the embedding, plus a skip (1×1 conv when channels change) → GELU"),
        (("down", "Down"), "Conv 3×3, stride 2 → BatchNorm → GELU"),
        (("up", "Up"), "ConvTranspose 4×4, stride 2 → BatchNorm → GELU"),
        ("attn", "Multi-head self-attention (4 heads) over the 64 positions of the 8×8 map, with GroupNorm and a residual add"),
        (("out", "Stem conv"), "First layer: Conv 3×3 → BatchNorm → GELU, 3 → 128 channels"),
        (("out", "Output conv"), "Last layer: Conv 3×3, 128 → 3 channels, giving the predicted flow"),
        ((f'<span class="unet-swatch" style="--c:{EMB}"></span>', "emb"),
         "FCB<sub>t</sub>(t) + FCB<sub>c</sub>(c), each 512-d, then GELU; each ResBlock maps it to a per-channel scale and shift"),
    ],
    note="Front face: spatial size. Depth: channels. D = 128; 21.0M parameters. Each block is a stage output, labelled with the operations that produced it. "
         "c is set to zero 10% of the time in training (p<sub>uncond</sub> = 0.1). Input shown: CIFAR-10 test image #3 mixed with noise at t = 0.5, for illustration.",
)


def build_resblock():
    """Flat flow diagram of one ResBlock from the wider/deeper CIFAR UNet."""
    p, RES, SK = "rb", OPS["res"][0], "#9aa0a6"
    Y, YS, YE = 126, 198, 30                          # main row, skip row, embedding row
    W, H = 860, 236
    def box(x0, x1, y, lines, col, fill, sub=None):
        out = [f'<rect x="{x0}" y="{y - 17}" width="{x1 - x0}" height="34" rx="8" style="fill:{fill}; stroke:{col}" class="rb-box"/>']
        out.append(f'<text class="rb-op" x="{(x0 + x1) / 2}" y="{y + 4.5}" text-anchor="middle" fill="{col}">{lines}</text>')
        return out
    def node(x, y, glyph, col, tip):
        g = (f'M{x - 3.6},{y - 3.6} L{x + 3.6},{y + 3.6} M{x + 3.6},{y - 3.6} L{x - 3.6},{y + 3.6}' if glyph == "x"
             else f'M{x - 4.4},{y} H{x + 4.4} M{x},{y - 4.4} V{y + 4.4}')
        return [f'<g class="unet-mul"><title>{tip}</title><circle cx="{x}" cy="{y}" r="10" style="stroke:{col}"/><path d="{g}" style="stroke:{col}"/></g>']
    def line(d, col, mk="ah", cls="unet-op"):
        return f'<path class="{cls}" style="stroke:{col}" d="{d}" marker-end="url(#{p}-{mk})"/>'
    parts = []
    # input / output tensors
    parts.append(f'<rect class="rb-io" x="14" y="{Y - 23}" width="90" height="46" rx="14"/>')
    parts.append(f'<text class="rb-io-t" x="59" y="{Y - 4}" text-anchor="middle"><tspan class="unet-sym">h</tspan><tspan class="unet-sub" dy="4">in</tspan></text>')
    parts.append(f'<text class="unet-dim" x="59" y="{Y + 15}" text-anchor="middle">C<tspan dy="3" font-size="9">in</tspan><tspan dy="-3">×H×W</tspan></text>')
    parts.append(f'<rect class="rb-io" x="758" y="{Y - 23}" width="92" height="46" rx="14"/>')
    parts.append(f'<text class="rb-io-t" x="804" y="{Y - 4}" text-anchor="middle"><tspan class="unet-sym">h</tspan><tspan class="unet-sub" dy="4">out</tspan></text>')
    parts.append(f'<text class="unet-dim" x="804" y="{Y + 15}" text-anchor="middle">C<tspan dy="3" font-size="9">out</tspan><tspan dy="-3">×H×W</tspan></text>')
    # main path
    parts += box(122, 300, Y, "Conv 3×3 → BN → GELU", RES, "#e8f5f6")
    parts += box(326, 452, Y, "Conv 3×3 → BN", RES, "#e8f5f6")
    xs, xa, xr = 494, 548, 624
    parts += node(xs, Y, "x", EMB, "Multiply each channel by (1 + scale)")
    parts += node(xa, Y, "+", EMB, "Add shift to each channel")
    parts += node(xr, Y, "+", "#6b7178", "Residual add: main path + skip")
    parts += box(656, 734, Y, "GELU", RES, "#e8f5f6")
    for x0, x1 in ((104, 122), (300, 326), (452, xs - 10), (xs + 10, xa - 10), (xa + 10, xr - 10), (xr + 10, 656), (734, 758)):
        parts.append(line(f"M{x0 + 2},{Y} H{x1 - 3}", "#6b7178"))
    # skip path
    parts.append(f'<path class="unet-skip" d="M59,{Y + 23} V{YS - 7} Q59,{YS} 66,{YS} H{xr - 7} Q{xr},{YS} {xr},{YS - 7} V{Y + 13}" marker-end="url(#{p}-ah-skip)"/>')
    parts.append(f'<rect class="rb-skipbox" x="250" y="{YS - 13}" width="236" height="26" rx="7"/>')
    parts.append(f'<text class="rb-skip-t" x="368" y="{YS + 4}" text-anchor="middle">1×1 Conv if C<tspan dy="3" font-size="9">in</tspan><tspan dy="-3"> ≠ C</tspan><tspan dy="3" font-size="9">out</tspan><tspan dy="-3">, else identity</tspan></text>')
    # conditioning from above
    parts.append(f'<rect class="unet-emb-pill" x="196" y="{YE - 12}" width="86" height="24" rx="12"/>')
    parts.append(f'<text class="unet-emb-text unet-emb-text--on" x="239" y="{YE + 4.5}" text-anchor="middle">emb · 512</text>')
    parts.append(line(f"M282,{YE} H{404}", EMB, "ah-emb", "unet-cond-line"))
    parts.append(f'<rect class="rb-lin" x="407" y="{YE - 14}" width="180" height="28" rx="7"/>')
    parts.append(f'<text class="rb-lin-t" x="497" y="{YE + 4.5}" text-anchor="middle">Linear 512 → 2·C<tspan dy="3" font-size="9">out</tspan></text>')
    # one 2·C_out output, split in half: first C_out values are the scales, last C_out the shifts
    xm, ysplit = (xs + xa) / 2, YE + 42
    parts.append(f'<path class="unet-cond-line" style="stroke:{EMB}" d="M{xm},{YE + 14} V{ysplit}"/>')
    parts.append(f'<text class="rb-lab" x="{xm + 7}" y="{YE + 31}">2·C<tspan dy="3" font-size="8.5">out</tspan></text>')
    parts.append(f'<rect class="rb-split" x="{xs - 2}" y="{ysplit - 1.5}" width="{xa - xs + 4}" height="3" rx="1.5"><title>Split the 2·C_out values in half</title></rect>')
    parts.append(f'<text class="rb-lab rb-lab--split" x="{xa + 10}" y="{ysplit + 4}">split</text>')
    for x in (xs, xa):
        parts.append(line(f"M{x},{ysplit + 1.5} V{Y - 13}", EMB, "ah-emb", "unet-cond-line"))
    parts.append(f'<text class="rb-lab" x="{xs - 7}" y="{ysplit + 25}" text-anchor="end">scale <tspan class="rb-lab-n">(first C<tspan dy="3" font-size="8.5">out</tspan><tspan dy="-3">)</tspan></tspan></text>')
    parts.append(f'<text class="rb-lab" x="{xa + 7}" y="{ysplit + 25}">shift <tspan class="rb-lab-n">(last C<tspan dy="3" font-size="8.5">out</tspan><tspan dy="-3">)</tspan></tspan></text>')
    parts.append(f'<text class="rb-lab rb-lab--node" x="{xs}" y="{Y + 28}" text-anchor="middle">× (1 + scale)</text>')
    parts.append(f'<text class="rb-lab rb-lab--node" x="{xa + 6}" y="{Y + 42}" text-anchor="middle">+ shift</text>')
    defs = (chevron(f"{p}-ah", "#6b7178") + chevron(f"{p}-ah-skip", SK) + chevron(f"{p}-ah-emb", EMB))
    svg = (f'<svg class="unet-svg rb-svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="rb-title rb-desc">\n'
           '<title id="rb-title">Residual block (ResBlock)</title>\n'
           '<desc id="rb-desc">The input goes through Conv 3×3, BatchNorm and GELU, then Conv 3×3 and BatchNorm. That result is multiplied channel-wise by '
           '(1 + scale) and shifted, where scale and shift come from a Linear layer mapping the 512-dimensional (t, c) embedding to 2 × C_out values. '
           'It is then added to the skip path (a 1×1 convolution when the channel count changes, otherwise the input itself) and passed through GELU.</desc>\n'
           f'<defs>{defs}</defs>\n' + "\n".join(parts) + "\n</svg>")
    return ('<figure class="unet-fig rb-fig">\n<figcaption class="rb-caption">ResBlock</figcaption>\n'
            '<div class="unet-scroll" tabindex="0" role="region" aria-label="Residual block diagram">\n' + svg + '\n</div>\n'
            '<p class="table-note">Each ResBlock has its own Linear layer, sized to its output channels. Its weights start at zero, '
            'so every block starts with scale 0 and shift 0, unconditioned.</p>\n</figure>')


def build_attnblock():
    """Flat flow diagram of the 8×8 self-attention layer in the wider/deeper bottleneck."""
    p, A = "ab", OPS["attn"][0]
    Y, YS = 62, 140
    W, H = 860, 168
    def box(x0, x1, label, dims=None, note=None):
        out = [f'<rect x="{x0}" y="{Y - 17}" width="{x1 - x0}" height="34" rx="8" class="rb-box" style="fill:#fdf3e2; stroke:{A}"/>',
               f'<text class="rb-op" x="{(x0 + x1) / 2}" y="{Y + 4.5}" text-anchor="middle" fill="{A}">{label}</text>']
        if dims:
            out.append(f'<text class="unet-dim" x="{(x0 + x1) / 2}" y="{Y + 32}" text-anchor="middle">{dims}</text>')
        if note:
            out.append(f'<text class="ab-note" x="{(x0 + x1) / 2}" y="{Y - 25}" text-anchor="middle">{note}</text>')
        return out
    def line(x0, x1):
        return f'<path class="unet-op" style="stroke:#6b7178" d="M{x0 + 2},{Y} H{x1 - 3}" marker-end="url(#{p}-ah)"/>'
    parts = []
    for x0, lab, sub, dims in ((14, "h", "in", "256×8×8"), (752, "h", "out", "256×8×8")):
        parts.append(f'<rect class="rb-io" x="{x0}" y="{Y - 23}" width="92" height="46" rx="14"/>')
        parts.append(f'<text class="rb-io-t" x="{x0 + 46}" y="{Y - 4}" text-anchor="middle"><tspan class="unet-sym">{lab}</tspan><tspan class="unet-sub" dy="4">{sub}</tspan></text>')
        parts.append(f'<text class="unet-dim" x="{x0 + 46}" y="{Y + 15}" text-anchor="middle">{dims}</text>')
    parts += box(128, 232, "GroupNorm", "8 groups")
    parts += box(254, 382, "Flatten", "64 tokens × 256")
    parts += box(404, 580, "Self-attention · 4 heads", "64 × 256")
    parts += box(602, 700, "Reshape", "256×8×8")
    xr = 726
    parts.append(f'<g class="unet-mul"><title>Residual add: input + attention output</title><circle cx="{xr}" cy="{Y}" r="10" style="stroke:#6b7178"/>'
                 f'<path d="M{xr - 4.4},{Y} H{xr + 4.4} M{xr},{Y - 4.4} V{Y + 4.4}" style="stroke:#6b7178"/></g>')
    for x0, x1 in ((106, 128), (232, 254), (382, 404), (580, 602), (700, xr - 10), (xr + 10, 752)):
        parts.append(line(x0, x1))
    parts.append(f'<path class="unet-skip" d="M60,{Y + 23} V{YS - 7} Q60,{YS} 67,{YS} H{xr - 7} Q{xr},{YS} {xr},{YS - 7} V{Y + 13}" marker-end="url(#{p}-ah-skip)"/>')
    parts.append(f'<rect class="rb-skipbox" x="300" y="{YS - 13}" width="206" height="26" rx="7"/>')
    parts.append(f'<text class="rb-skip-t" x="403" y="{YS + 4}" text-anchor="middle">identity (un-normalised input)</text>')
    defs = chevron(f"{p}-ah", "#6b7178") + chevron(f"{p}-ah-skip", "#9aa0a6")
    svg = (f'<svg class="unet-svg rb-svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="ab-title ab-desc">\n'
           '<title id="ab-title">Self-attention layer (Attn)</title>\n'
           '<desc id="ab-desc">The 256×8×8 input is group-normalised, flattened into 64 tokens of 256 channels, passed through 4-head self-attention '
           'with queries, keys and values all projected from the tokens, reshaped back to 256×8×8 and added to the un-normalised input.</desc>\n'
           f'<defs>{defs}</defs>\n' + "\n".join(parts) + "\n</svg>")
    return ('<figure class="unet-fig rb-fig">\n<figcaption class="rb-caption" style="color:#c77d12">Attn</figcaption>\n'
            '<div class="unet-scroll" tabindex="0" role="region" aria-label="Self-attention diagram">\n' + svg + '\n</div>\n</figure>')


def splice(s, prefix, figure, anchor, heading):
    start, end = f"<!-- {prefix}-diagram:start -->", f"<!-- {prefix}-diagram:end -->"
    block = f"{start}\n{figure}\n{end}"
    if start in s:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda m: block, s, flags=re.S)
    assert s.count(anchor) == 1, anchor
    return s.replace(anchor, heading + "\n" + block + "\n" + anchor)


if __name__ == "__main__":
    s = PAGE.read_text()
    s = splice(s, "unet", build(UNCOND), '<h1 class="sub-after-intro" id="b-denoise">', '<h1 id="b-unet">1.1 Implementing the UNet</h1>')
    s = splice(s, "tunet", build(TIME), '<h1>2.2–2.3 Time-Conditioned UNet</h1>', '<h1 id="b-tunet">2.1 Adding Time Conditioning to UNet</h1>')
    s = splice(s, "dunet", build(DEEP), '<p class="exif params"><span>Parameters <b>21.0M</b></span>', '')
    s = splice(s, "resblock", build_resblock(), '<p class="exif params"><span>Parameters <b>21.0M</b></span>', '')
    s = splice(s, "attnblock", build_attnblock(), '<p class="exif params"><span>Parameters <b>21.0M</b></span>', '')
    s = splice(s, "cunet", build(CLASS), '<h1 id="b-class-train">2.5 Training the UNet</h1>', '<h1 id="b-cunet">2.4 Adding Class-Conditioning to UNet</h1>')
    PAGE.write_text(s)
