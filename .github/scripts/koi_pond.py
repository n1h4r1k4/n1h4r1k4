"""Put git-koi's koi in a wide, animated, anime-style pond for a profile README.

git-koi draws the koi on a clear background. This draws a pond around it:
cel-shaded water with drifting caustics and light shafts, raindrop ripples, a
wake behind the koi and its shadow, lily pads and lotus flowers it swims under,
and twinkling glints. The pond is drawn at the koi's own pixel size and then
doubled, like the koi, and loops with it.

Usage: python koi_pond.py koi/koi.gif koi/pond.gif
"""

import sys

import numpy as np
from PIL import Image, ImageSequence

TAU = 2 * np.pi

# Water from deepest to brightest; every water effect moves a pixel along it.
WATER = [
    (9, 28, 60), (13, 42, 84), (17, 58, 106), (21, 76, 126), (25, 95, 145), (31, 115, 162),
    (42, 137, 177), (61, 160, 191), (94, 184, 207), (138, 209, 223), (189, 232, 240), (238, 251, 253),
]
PAD_OUTLINE, PAD_SHADE, PAD_BASE, PAD_LIGHT, PAD_SHINE = (
    (20, 66, 44), (46, 112, 56), (70, 146, 66), (116, 182, 82), (176, 218, 116))
LOTUS_OUTLINE, LOTUS_PETAL, LOTUS_INNER, LOTUS_HEART = (
    (170, 60, 108), (238, 134, 172), (255, 202, 220), (252, 214, 90))
FIXED_COLORS = WATER + [PAD_OUTLINE, PAD_SHADE, PAD_BASE, PAD_LIGHT, PAD_SHINE,
                        LOTUS_OUTLINE, LOTUS_PETAL, LOTUS_INNER, LOTUS_HEART]
PAD_INDEX = {color: 1 + len(WATER) + i for i, color in enumerate(FIXED_COLORS[len(WATER):])}

# Lily pads as (x, y, radius, notch angle, has a lotus), in pond pixels on the
# 512 x 256 pond; the koi's square swims in the middle, x 128 to 384.
PADS = [
    (44, 56, 23, 0.7, True), (100, 150, 15, 2.5, False), (34, 206, 19, 4.1, False),
    (131, 92, 11, 5.4, False), (470, 66, 20, 3.5, False), (436, 194, 25, 1.1, True),
    (386, 132, 12, 0.3, False), (500, 150, 9, 2.0, False),
]
# Raindrops as (x, y, phase): each falls once per loop and rings out.
DROPS = [(176, 56, 0.00), (300, 210, 0.14), (70, 112, 0.29), (452, 130, 0.43),
         (236, 150, 0.57), (150, 230, 0.71), (360, 36, 0.86)]


def shift(mask, dx, dy):
    """Move a 2-D array by whole pixels, filling the uncovered edge with zeros."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        mask[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def edge_of(mask):
    """Pixels of a mask that touch a pixel outside it."""
    return mask & ~(shift(mask, 1, 0) & shift(mask, -1, 0) & shift(mask, 0, 1) & shift(mask, 0, -1))


def main(source, target):
    koi_frames, durations = [], []
    for frame in ImageSequence.Iterator(Image.open(source)):
        rgba = np.asarray(frame.convert("RGBA"))
        small = rgba[::2, ::2]
        if not np.array_equal(small.repeat(2, 0).repeat(2, 1), rgba):
            sys.exit("the koi GIF is not drawn at twice the koi's pixel size")
        koi_frames.append(small)
        durations.append(frame.info.get("duration", 60))
    count = len(koi_frames)
    kh, kw = koi_frames[0].shape[:2]
    ah, aw = kh, 2 * kh
    ox = (aw - kw) // 2
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float64)

    # Palette: 0 is clear, then the pond's colors, then every color the koi uses.
    def keys_of(rgb):
        return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    koi_keys = [keys_of(f) for f in koi_frames]
    koi_opaque = [f[..., 3] > 0 for f in koi_frames]
    koi_colors = np.unique(np.concatenate([k[o] for k, o in zip(koi_keys, koi_opaque)]))
    koi_base = 1 + len(FIXED_COLORS)
    if koi_base + len(koi_colors) > 256:
        sys.exit(f"the koi uses {len(koi_colors)} colors; the pond GIF has room for {256 - koi_base}")
    koi_rgb = [((k >> 16) & 255, (k >> 8) & 255, k & 255) for k in koi_colors.tolist()]
    palette = [c for color in [(0, 0, 0)] + FIXED_COLORS + koi_rgb for c in color]

    # The pond's outline, a rounded rectangle, and how far each pixel is from its rim.
    rc = 12
    cx = np.clip(xx, rc, aw - 1 - rc)
    cy = np.clip(yy, rc, ah - 1 - rc)
    rim_distance = np.hypot(xx - cx, yy - cy)
    inside = rim_distance <= rc + 0.3
    border = inside & (rim_distance > rc - 0.7)
    corner = ((xx < rc) | (xx > aw - 1 - rc)) & ((yy < rc) | (yy > ah - 1 - rc))
    edge = np.where(corner, rc - rim_distance, np.minimum.reduce([xx, aw - 1 - xx, yy, ah - 1 - yy]))

    # Depth: bright shallows along the rim, darkening toward the middle.
    edge = edge + 3 * np.sin(xx / 19 + 1.3) * np.sin(yy / 13) + 2 * np.sin((xx + yy) / 11) \
        + 6 * np.sin(xx / 47 + 0.4) * np.sin(yy / 31 + 1.1)
    deep = edge + 16 * np.sin(xx / 53 + 2.2) * np.sin(yy / 37 + 0.5) + 10 * np.sin((xx - 1.7 * yy) / 61)
    depth = np.select([edge < 4, edge < 10, edge < 22, deep < 56], [7, 6, 5, 4], 3).astype(np.int16)

    # Caustics: the edges of a gently swirling, warped Voronoi pattern.
    rng = np.random.default_rng(7)
    cell, margin = 24, 3
    ncy, ncx = ah // cell + 2 * margin + 1, aw // cell + 2 * margin + 1
    gj, gi = np.mgrid[0:ncy, 0:ncx]
    seed_x = (gi - margin + rng.uniform(0.2, 0.8, (ncy, ncx))) * cell
    seed_y = (gj - margin + rng.uniform(0.2, 0.8, (ncy, ncx))) * cell
    orbit = rng.uniform(3, 6, (ncy, ncx))
    turns = rng.choice([-2, -1, 1, 2], (ncy, ncx))
    phase = rng.uniform(0, TAU, (ncy, ncx))

    def caustics(t):
        px = seed_x + orbit * np.cos(TAU * turns * t + phase)
        py = seed_y + orbit * np.sin(TAU * turns * t + phase)
        wx = xx + 3.2 * np.sin(yy / 9 + TAU * t) + 1.5 * np.sin(yy / 23 - TAU * t + 0.6)
        wy = yy + 3.2 * np.sin(xx / 13 + TAU * t + 1.7) + 1.5 * np.sin(xx / 29 + TAU * t + 2.9)
        ci = np.floor(wx / cell).astype(int) + margin
        cj = np.floor(wy / cell).astype(int) + margin
        f1 = np.full(xx.shape, np.inf)
        f2 = np.full(xx.shape, np.inf)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                d = (px[cj + dj, ci + di] - wx) ** 2 + (py[cj + dj, ci + di] - wy) ** 2
                f2 = np.where(d < f1, f1, np.minimum(f2, d))
                f1 = np.minimum(f1, d)
        gap = np.sqrt(f2) - np.sqrt(f1)
        # Sunlight falls in drifting dapples: caustics glow inside them, fade outside.
        dapple = np.sin(xx / 41 + TAU * t) * np.sin(yy / 29 - TAU * t + 0.8) \
            + 0.6 * np.sin((xx + 2 * yy) / 53 + TAU * t + 2.0) + 0.25 * (depth - 4)
        line, halo = gap < 1.1, gap < 2.4
        return np.select([line & (dapple > 0.25), (line & (dapple > -0.45)) | (halo & (dapple > 0.55))],
                         [2, 1], 0)

    def rings(delta, x, y, radii, strength):
        reach = max(radii) + 2
        x0, x1 = int(max(x - reach, 0)), int(min(x + reach + 1, aw))
        y0, y1 = int(max(y - reach, 0)), int(min(y + reach + 1, ah))
        if x0 >= x1 or y0 >= y1:
            return
        dx, dy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
        d = np.hypot(dx, dy)
        lit = dx * 0.6 + dy * 0.8 < 0
        area = delta[y0:y1, x0:x1]
        for r in radii:
            if r > 0.8:
                on = np.abs(d - r) < 0.7
                area[on & lit] = np.maximum(area[on & lit], strength)
                area[on & ~lit] = np.maximum(area[on & ~lit], max(strength - 2, 1))

    # Lily pads and lotus flowers, drawn once, and the shadow they cast.
    pads = np.zeros((ah, aw), np.uint8)
    for px_, py_, r, notch, lotus in PADS:
        dx, dy = xx - px_, yy - py_
        d, angle = np.hypot(dx, dy), np.arctan2(dy, dx)
        gap = np.abs((angle - notch + np.pi) % TAU - np.pi) < 0.34
        body = (d <= r) & ~(gap & (d > 1.0))
        spokes = np.abs((((angle - notch) * 7 / TAU) % 1) - 0.5) * TAU / 7 * d < 0.5
        layer = np.where(np.hypot(dx + 2, dy + 2.5) > r - 0.5, PAD_INDEX[PAD_SHADE], PAD_INDEX[PAD_BASE])
        layer = np.where(spokes & (d > 2.5) & (d < r - 2), PAD_INDEX[PAD_SHADE], layer)
        layer = np.where(np.hypot(dx - 2, dy - 2.5) > r - 0.5, PAD_INDEX[PAD_LIGHT], layer)
        layer = np.where((np.hypot(dx - 1, dy - 1.3) > r - 1.6) & (dx * 0.6 + dy * 0.8 < -0.55 * d),
                         PAD_INDEX[PAD_SHINE], layer)
        layer = np.where(edge_of(body), PAD_INDEX[PAD_OUTLINE], layer)
        pads = np.where(body, layer, pads).astype(np.uint8)
        if lotus:
            fx, fy = px_ - 0.25 * r * np.cos(notch), py_ - 0.25 * r * np.sin(notch)
            dx, dy = xx - fx, yy - fy
            d, angle = np.hypot(dx, dy), np.arctan2(dy, dx)
            outer = d <= 5.2 + 1.8 * np.cos(5 * angle)
            inner = d <= 2.9 + 1.0 * np.cos(5 * angle + np.pi / 5)
            flower = np.where(inner, PAD_INDEX[LOTUS_INNER], PAD_INDEX[LOTUS_PETAL])
            flower = np.where(d <= 1.2, PAD_INDEX[LOTUS_HEART], flower)
            flower = np.where(edge_of(outer), PAD_INDEX[LOTUS_OUTLINE], flower)
            pads = np.where(outer, flower, pads).astype(np.uint8)
    pad_shadow = shift(pads > 0, 2, 3)

    glints = np.random.default_rng(11)
    glint_xy = glints.uniform([10, 10], [aw - 10, ah - 10], (18, 2)).astype(int)
    glint_phase = glints.uniform(0, 1, 18)

    # Where the koi is in each frame, for its wake and its shadow.
    centers = []
    for opaque in koi_opaque:
        ys, xs = np.nonzero(opaque)
        centers.append((xs.mean() + ox, ys.mean()) if len(xs) else None)

    frames, previous = [], None
    for f in range(count):
        s = f - f % 2  # the water moves on twos, the koi on ones
        t = s / count
        level = depth.copy()
        level += (((xx * 0.85 + yy * 0.5) / 150 - t) % 1 < 0.12).astype(np.int16)
        level += caustics(t).astype(np.int16)
        delta = np.zeros((ah, aw), np.int16)
        for x, y, drop_phase in DROPS:
            age = ((t + drop_phase) % 1) / 0.6
            if age < 1:
                r = 1.5 + age * 30
                rings(delta, x, y, [r, r - 5, r - 10], 4 if age < 0.3 else 3 if age < 0.65 else 2)
                if age < 0.06:
                    delta[y, x - 1:x + 2] = 4
                    delta[y - 1:y + 2, x] = 4
        for spawn in range(0, count, 10):
            age = (s - spawn) % count
            if age < 30 and centers[spawn] is not None:
                r = 2 + age * 0.8
                rings(delta, *centers[spawn], [r, r - 4], 3 if age < 12 else 2)
        level += delta
        koi_shadow = np.zeros((ah, aw), bool)
        koi_shadow[:, ox:ox + kw] = shift(koi_opaque[f], 3, 4)
        level -= 3 * koi_shadow + 2 * pad_shadow
        level = np.clip(level, 0, len(WATER) - 1)
        for (gx, gy), glint in zip(glint_xy, glint_phase):
            age = (t + glint) % 1
            if age < 0.07:
                arm = 2 if 0.02 <= age < 0.05 else 1
                level[gy, gx - arm:gx + arm + 1] = np.maximum(level[gy, gx - arm:gx + arm + 1], 10)
                level[gy - arm:gy + arm + 1, gx] = np.maximum(level[gy - arm:gy + arm + 1, gx], 10)
                level[gy, gx] = 11
        index = (1 + level).astype(np.uint8)
        koi_index = (koi_base + np.searchsorted(koi_colors, koi_keys[f])).astype(np.uint8)
        view = index[:, ox:ox + kw]
        view[koi_opaque[f]] = koi_index[koi_opaque[f]]
        index = np.where(pads > 0, pads, index)
        index[border] = 1
        index[~inside] = 0
        full = index.repeat(2, 0).repeat(2, 1)
        # After the first frame, pixels that did not change are left clear, so
        # each frame stores only what moved.
        stored = full if previous is None else np.where(full == previous, 0, full).astype(np.uint8)
        previous = full
        image = Image.fromarray(stored, "P")
        image.putpalette(palette)
        frames.append(image)

    frames[0].save(target, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=1, transparency=0, optimize=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
