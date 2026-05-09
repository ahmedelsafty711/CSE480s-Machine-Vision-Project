"""
Dataset Generator
=================
Creates a synthetic 6-class image dataset using the minicv drawing library.

Classes
-------
circle, square, triangle, star, cross, hexagon

Each image is 64×64 RGB with:
- Random background color + subtle gradient
- Shape at random position, size, rotation, color
- Gaussian noise for realism
- Intra-class variability: lighting, background, viewpoint/scale/pose

Output
------
- milestone2/dataset/images/<class>/<id>.png
- milestone2/dataset/annotations.csv  (filepath, label, split)
"""

from __future__ import annotations
import sys, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import minicv as cv

# ── Configuration ──────────────────────────────────────────────────────────
CLASSES      = ["circle", "square", "triangle", "star", "cross", "hexagon"]
N_PER_CLASS  = 220       # 154 train / 33 val / 33 test  (≈70/15/15)
IMG_SIZE     = 64
SEED         = 42
OUT_DIR      = os.path.join(os.path.dirname(__file__), "images")
ANN_FILE     = os.path.join(os.path.dirname(__file__), "annotations.csv")

rng = np.random.default_rng(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────

def _rand_color():
    """Random vivid RGB color, avoiding very dark/light shades."""
    h = rng.uniform(0, 360)
    s = rng.uniform(0.5, 1.0)
    v = rng.uniform(0.5, 1.0)
    # HSV → RGB
    hi = int(h / 60) % 6
    f  = h / 60 - int(h / 60)
    p, q, t = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    rgb_map = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
    r, g, b = rgb_map[hi]
    return (int(r*255), int(g*255), int(b*255))


def _make_background(size: int) -> np.ndarray:
    """Random solid or gradient background with lighting variation."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    base = _rand_color()
    # Lighting offset: simulate different ambient conditions
    brightness = rng.uniform(0.3, 1.0)
    for c in range(3):
        canvas[:, :, c] = int(base[c] * brightness)
    # Random gradient direction for background variation
    if rng.random() > 0.5:
        direction = rng.integers(0, 4)
        grad = np.linspace(0, rng.integers(30, 80), size)
        if direction == 0:   # left→right
            for x in range(size):
                canvas[:, x, :] = np.clip(canvas[:, x, :] + int(grad[x]), 0, 255)
        elif direction == 1: # top→bottom
            for y in range(size):
                canvas[y, :, :] = np.clip(canvas[y, :, :] + int(grad[y]), 0, 255)
    return canvas


def _add_noise(canvas: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    noise = rng.normal(0, sigma, canvas.shape)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _polygon_rotated(cx, cy, n_sides, radius, angle_offset):
    """Generate n-sided polygon vertices centered at (cx, cy) with rotation."""
    verts = []
    for i in range(n_sides):
        a = 2 * math.pi * i / n_sides + angle_offset
        verts.append((int(cx + radius * math.cos(a)),
                      int(cy + radius * math.sin(a))))
    return verts


def _star_vertices(cx, cy, outer_r, inner_r, n_points, angle_offset):
    """Generate n-pointed star vertices."""
    verts = []
    for i in range(2 * n_points):
        r = outer_r if i % 2 == 0 else inner_r
        a = math.pi * i / n_points + angle_offset
        verts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
    return verts


# ── Per-class drawing functions ────────────────────────────────────────────

def draw_circle(canvas, size):
    cx = rng.integers(15, size - 15)
    cy = rng.integers(15, size - 15)
    r  = rng.integers(8, min(cx, cy, size-cx, size-cy) - 2)
    color = _rand_color()
    # Draw filled circle via polygon approximation (32 vertices)
    verts = _polygon_rotated(cx, cy, 32, r, 0.0)
    cv.draw_polygon(canvas, verts, color=color, filled=True)
    # Optional outline
    if rng.random() > 0.5:
        outline = tuple(max(0, c - 50) for c in color)
        cv.draw_polygon(canvas, verts, color=outline, filled=False, thickness=1)


def draw_square(canvas, size):
    side = rng.integers(14, 34)
    cx   = rng.integers(side//2 + 5, size - side//2 - 5)
    cy   = rng.integers(side//2 + 5, size - side//2 - 5)
    angle = rng.uniform(0, math.pi / 4)   # rotation: viewpoint variation
    verts = _polygon_rotated(cx, cy, 4, side // 2 * math.sqrt(2), angle + math.pi/4)
    color = _rand_color()
    cv.draw_polygon(canvas, verts, color=color, filled=True)
    if rng.random() > 0.5:
        outline = tuple(max(0, c - 60) for c in color)
        cv.draw_polygon(canvas, verts, color=outline, filled=False, thickness=1)


def draw_triangle(canvas, size):
    cx = rng.integers(12, size - 12)
    cy = rng.integers(12, size - 12)
    r  = rng.integers(10, 24)
    angle_offset = rng.uniform(0, 2 * math.pi)   # rotation variability
    verts = _polygon_rotated(cx, cy, 3, r, angle_offset)
    color = _rand_color()
    cv.draw_polygon(canvas, verts, color=color, filled=True)
    if rng.random() > 0.5:
        outline = tuple(max(0, c - 60) for c in color)
        cv.draw_polygon(canvas, verts, color=outline, filled=False, thickness=1)


def draw_star(canvas, size):
    cx = rng.integers(18, size - 18)
    cy = rng.integers(18, size - 18)
    outer_r = rng.integers(12, 22)
    inner_r = max(4, int(outer_r * rng.uniform(0.35, 0.55)))
    angle_offset = rng.uniform(0, 2 * math.pi)
    n_points = 5
    verts = _star_vertices(cx, cy, outer_r, inner_r, n_points, angle_offset - math.pi/2)
    color = _rand_color()
    cv.draw_polygon(canvas, verts, color=color, filled=True)
    if rng.random() > 0.5:
        outline = tuple(max(0, c - 60) for c in color)
        cv.draw_polygon(canvas, verts, color=outline, filled=False, thickness=1)


def draw_cross(canvas, size):
    cx = rng.integers(14, size - 14)
    cy = rng.integers(14, size - 14)
    arm_len  = rng.integers(10, 22)
    arm_w    = rng.integers(3, 8)
    angle    = rng.uniform(-0.3, 0.3)   # slight rotation
    color    = _rand_color()
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Cross as two overlapping rectangles approximated by filled polygons
    def rotated_rect(cx, cy, hw, hh):
        pts = [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
        return [(int(cx + p[0]*cos_a - p[1]*sin_a),
                 int(cy + p[0]*sin_a + p[1]*cos_a)) for p in pts]
    h_bar = rotated_rect(cx, cy, arm_len, arm_w)
    v_bar = rotated_rect(cx, cy, arm_w, arm_len)
    cv.draw_polygon(canvas, h_bar, color=color, filled=True)
    cv.draw_polygon(canvas, v_bar, color=color, filled=True)


def draw_hexagon(canvas, size):
    cx = rng.integers(14, size - 14)
    cy = rng.integers(14, size - 14)
    r  = rng.integers(10, 22)
    angle_offset = rng.uniform(0, math.pi / 6)
    verts = _polygon_rotated(cx, cy, 6, r, angle_offset)
    color = _rand_color()
    cv.draw_polygon(canvas, verts, color=color, filled=True)
    if rng.random() > 0.5:
        outline = tuple(max(0, c - 60) for c in color)
        cv.draw_polygon(canvas, verts, color=outline, filled=False, thickness=1)


DRAW_FUNCS = {
    "circle":   draw_circle,
    "square":   draw_square,
    "triangle": draw_triangle,
    "star":     draw_star,
    "cross":    draw_cross,
    "hexagon":  draw_hexagon,
}

# ── Main generation loop ───────────────────────────────────────────────────

def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []

    for cls in CLASSES:
        cls_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print(f"  Generating {N_PER_CLASS} images for class '{cls}' …")

        for i in range(N_PER_CLASS):
            canvas = _make_background(IMG_SIZE)
            DRAW_FUNCS[cls](canvas, IMG_SIZE)
            canvas = _add_noise(canvas, sigma=rng.uniform(5, 18))

            fname = f"{cls}_{i:04d}.png"
            fpath = os.path.join(cls_dir, fname)
            plt.imsave(fpath, canvas)
            records.append((os.path.join("images", cls, fname), cls))

    # Shuffle and split train/val/test per class (stratified)
    rng_split = np.random.default_rng(99)
    rows = []
    for cls in CLASSES:
        cls_records = [(p, l) for p, l in records if l == cls]
        idx = rng_split.permutation(len(cls_records))
        n_train = int(0.70 * len(cls_records))
        n_val   = int(0.15 * len(cls_records))
        for j, k in enumerate(idx):
            split = "train" if j < n_train else "val" if j < n_train + n_val else "test"
            rows.append((cls_records[k][0], cls_records[k][1], split))

    import csv
    with open(ANN_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "split"])
        w.writerows(rows)

    counts = {}
    for _, _, s in rows:
        counts[s] = counts.get(s, 0) + 1
    print(f"\n  Saved {len(rows)} images → {ANN_FILE}")
    print(f"  Splits: {counts}")

    # Class distribution plot
    import collections
    label_counts = collections.Counter(l for _, l, _ in rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(label_counts.keys(), label_counts.values(), color="#4cc9f0")
    ax.set_title("Class Distribution", fontsize=13)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "class_distribution.png")
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print(f"  Distribution plot → {plot_path}")


if __name__ == "__main__":
    print("Generating dataset …")
    generate()
    print("Done.")
