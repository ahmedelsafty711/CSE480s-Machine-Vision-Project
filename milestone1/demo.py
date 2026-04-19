"""
minicv Milestone 1 — Live Demo
================================
Run with:
    python milestone1/demo.py

Produces:
  - Printed results in the terminal  (pass/fail for every feature)
  - milestone1/demo_output.png        (all visual results in one figure)

No real image files needed — everything uses a synthetic test image.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import minicv as cv

# ── colour palette for pass/fail output ──────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"

_results = []

def check(name, condition):
    status = f"{GREEN}✓ PASS{RESET}" if condition else f"{RED}✗ FAIL{RESET}"
    print(f"  {status}  {name}")
    _results.append((name, condition))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD A SYNTHETIC "NATURAL-LOOKING" TEST IMAGE
# ─────────────────────────────────────────────────────────────────────────────
img_rgb = cv.read_image(r"D:\ASU\spring 26\vision\project\download (4).jpg")   # or .jpg
H, W = img_rgb.shape[:2]

img_gray_f = cv.rgb_to_gray(img_rgb)
img_gray   = (img_gray_f * 255).astype(np.float32)

# recreate the salt-and-pepper version from your image
rng = np.random.default_rng(42)
img_sp = img_rgb.copy()
sp_mask = rng.random((H, W)) < 0.05
img_sp[sp_mask] = 255
sp_mask2 = rng.random((H, W)) < 0.05
img_sp[sp_mask2] = 0
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  minicv Milestone 1 — Feature Demo{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

# ─────────────────────────────────────────────────────────────────────────────
print(f"{BOLD}[2] Image I/O & Color Conversion{RESET}")
gray = cv.rgb_to_gray(img_rgb)
check("rgb_to_gray → shape (H,W)",       gray.shape == (H, W))
check("rgb_to_gray → float32 in [0,1]",  gray.dtype == np.float32 and gray.max() <= 1.0)
rgb_back = cv.gray_to_rgb(gray)
check("gray_to_rgb → shape (H,W,3)",     rgb_back.shape == (H, W, 3))
check("gray_to_rgb → uint8",             rgb_back.dtype == np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}[3] Core Operations{RESET}")

n_mm  = cv.normalize(img_gray, mode="minmax")
check("normalize minmax → [0,1]",        n_mm.min() >= 0 and n_mm.max() <= 1.0)
n_zs  = cv.normalize(img_gray, mode="zscore")
check("normalize zscore → mean≈0",       abs(float(n_zs.mean())) < 0.1)
n_fx  = cv.normalize(img_gray, mode="fixed", fixed_min=0, fixed_max=255)
check("normalize fixed  → [0,1]",        n_fx.min() >= 0 and n_fx.max() <= 1.0)

clipped = cv.clip_pixels(img_gray - 50, 0, 255)
check("clip_pixels → no values < 0",     clipped.min() >= 0)

pad_z = cv.pad_image(img_gray, 10, 10, mode="zero")
check("pad zero     → shape (H+20,W+20)",pad_z.shape == (H+20, W+20))
pad_r = cv.pad_image(img_gray, 10, 10, mode="reflect")
check("pad reflect  → shape (H+20,W+20)",pad_r.shape == (H+20, W+20))
pad_e = cv.pad_image(img_gray, 10, 10, mode="replicate")
check("pad replicate→ shape (H+20,W+20)",pad_e.shape == (H+20, W+20))

identity = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
conv_out = cv.convolve2d(img_gray, identity)
check("convolve2d identity → same shape", conv_out.shape == img_gray.shape)
check("convolve2d identity → same values (interior)",
      np.allclose(conv_out[2:-2,2:-2], img_gray[2:-2,2:-2], atol=1))

spat = cv.spatial_filter(img_rgb.astype(np.float32),
                          np.ones((3,3),dtype=np.float32)/9)
check("spatial_filter RGB → shape (H,W,3)", spat.shape == (H,W,3))

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}[4] Image Processing Techniques{RESET}")

mean_out = cv.mean_filter(img_gray, ksize=9)
check("mean_filter   → smooths (std drops)",
      float(mean_out.std()) < float(img_gray.std()))

gk = cv.gaussian_kernel(5, 1.0)
check("gaussian_kernel 5×5 → sums to 1",  abs(gk.sum() - 1.0) < 1e-5)
gauss_out = cv.gaussian_filter(img_gray, ksize=5, sigma=1.5)
check("gaussian_filter → same shape",      gauss_out.shape == img_gray.shape)

median_out = cv.median_filter(
    img_sp[:,:,0].astype(np.float32), ksize=3)
sp_pixels  = int(np.sum(img_sp[:,:,0] == 255))
rem_pixels = int(np.sum(median_out > 250))
check("median_filter → removes salt&pepper",  rem_pixels < sp_pixels // 2)

bin_global = cv.threshold_global(img_gray, thresh=128)
vals_g = set(np.unique(bin_global).tolist())
check("threshold_global → only {0, 255}",  vals_g.issubset({0.0, 255.0}))

bin_otsu, t = cv.threshold_otsu(img_gray)
check(f"threshold_otsu → t={t:.1f}, binary",
      0 < t < 255 and set(np.unique(bin_otsu).tolist()).issubset({0.0,255.0}))

bin_adapt = cv.threshold_adaptive(img_gray, block_size=21, method="gaussian")
check("threshold_adaptive → only {0, 255}",
      set(np.unique(bin_adapt).tolist()).issubset({0.0,255.0}))

sobel = cv.sobel_gradients(img_gray)
check("sobel → magnitude non-negative",    sobel["magnitude"].min() >= 0)
check("sobel → angle in [0°,180°]",
      sobel["angle"].min() >= 0 and sobel["angle"].max() <= 180)

plane7 = cv.bit_plane_slice(img_gray, bit=7)
check("bit_plane_slice MSB → only {0,255}",
      set(np.unique(plane7).tolist()).issubset({0,255}))

counts, edges = cv.histogram(img_gray)
check("histogram → 256 bins summing to N", 
      len(counts) == 256 and int(counts.sum()) == img_gray.size)

eq = cv.histogram_equalization(img_gray)
check("histogram_equalization → uint8 [0,255]",
      eq.dtype == np.uint8 and eq.max() <= 255)

sharp = cv.unsharp_mask(img_gray, amount=1.5)
check("unsharp_mask (additional 1) → same shape", sharp.shape == img_gray.shape)

eroded  = cv.morphological_op(bin_global, op="erode")
dilated = cv.morphological_op(bin_global, op="dilate")
check("morphological erode  → shrinks foreground", eroded.sum()  <= bin_global.sum())
check("morphological dilate → grows  foreground",  dilated.sum() >= bin_global.sum())

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}[5] Geometric Transforms{RESET}")

resized_nn = cv.resize(img_rgb, 128, 128, interpolation="nearest")
check("resize nearest    → (128,128,3)", resized_nn.shape == (128,128,3))
resized_bl = cv.resize(img_rgb, 64, 64,  interpolation="bilinear")
check("resize bilinear   → (64,64,3)",   resized_bl.shape == (64,64,3))

rotated = cv.rotate(img_rgb, angle=30)
check("rotate 30°  → same shape",        rotated.shape == img_rgb.shape)
rotated180 = cv.rotate(img_gray, angle=180)
check("rotate 180° → same shape (gray)", rotated180.shape == img_gray.shape)

translated = cv.translate(img_rgb, tx=20, ty=20)
check("translate (20,20) → same shape",  translated.shape == img_rgb.shape)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}[6] Feature Extractors{RESET}")

ch = cv.color_histogram(img_rgb, bins=32)
check("color_histogram RGB → length 96",  ch.shape == (96,))

hu = cv.hu_moments(img_gray)
check("hu_moments → 7 values",            hu.shape == (7,))

hog_feat = cv.hog(img_gray, cell_size=8, block_size=2, n_bins=9)
check("hog → 1-D vector (non-empty)",     hog_feat.ndim == 1 and len(hog_feat) > 0)
print(f"      HOG feature length: {len(hog_feat)}")

gh = cv.gradient_hist(img_gray, bins=32)
check("gradient_hist → length 32",        gh.shape == (32,))

lbp_feat = cv.lbp(img_gray, radius=1, n_points=8, bins=32)
check("lbp → length 32",                  lbp_feat.shape == (32,))

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}[7+8] Drawing & Text{RESET}")

canvas = img_rgb.copy()
cv.draw_point(canvas,     x=128, y=30,  color=(255,255,0), radius=8)
cv.draw_line(canvas,      x0=10, y0=10, x1=245, y1=10, color=(255,0,0),   thickness=2)
cv.draw_rectangle(canvas, x0=10, y0=20, x1=245, y1=235,
                  color=(0,255,0), thickness=2, filled=False)
cv.draw_polygon(canvas,
                vertices=[(128,50),(180,150),(76,150)],
                color=(255,165,0), thickness=2, filled=False)
cv.put_text(canvas, "MINICV DEMO", x=12, y=200, font_scale=2, color=(255,255,255))

check("draw_point     → modifies canvas",  canvas[30,128,0] > 0 or canvas[30,128,1] > 0)
check("draw_line      → top row changed",  canvas[10,128,0] == 255)
check("put_text       → pixels written",   canvas[200:215, 12:100].sum() > 0)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
total  = len(_results)
passed = sum(1 for _, ok in _results if ok)
failed = total - passed

print(f"\n{BOLD}{'='*60}{RESET}")
if failed == 0:
    print(f"{GREEN}{BOLD}  ALL {total} CHECKS PASSED ✓{RESET}")
else:
    print(f"{RED}{BOLD}  {passed}/{total} PASSED — {failed} FAILED{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL OUTPUT — one big figure with all results
# ─────────────────────────────────────────────────────────────────────────────
print("Generating visual output …")

fig = plt.figure(figsize=(22, 30), facecolor="#1a1a2e")
fig.suptitle("minicv — Milestone 1 Feature Demo", fontsize=22,
             color="white", fontweight="bold", y=0.98)

def _ax(fig, gs, row, col, colspan=1):
    return fig.add_subplot(gs[row, col:col+colspan])

def show(ax, img, title, cmap=None):
    ax.set_facecolor("#16213e")
    if img.ndim == 3:
        ax.imshow(img.astype(np.uint8))
    else:
        ax.imshow(img, cmap=cmap or "gray", vmin=0)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.axis("off")

def show_hist(ax, data, title, color="#4cc9f0"):
    ax.set_facecolor("#16213e")
    ax.bar(range(len(data)), data, color=color, width=1.0)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

gs = gridspec.GridSpec(8, 5, figure=fig,
                       hspace=0.45, wspace=0.25,
                       left=0.04, right=0.97,
                       top=0.95, bottom=0.02)

# Row 0 — source images
show(_ax(fig,gs,0,0), img_rgb,                    "Original RGB")
show(_ax(fig,gs,0,1), (gray*255).astype(np.uint8),"Grayscale")
show(_ax(fig,gs,0,2), img_sp,                     "Salt & Pepper noise")
show(_ax(fig,gs,0,3), resized_bl.astype(np.uint8),"Resize 64×64 (bilinear)")
show(_ax(fig,gs,0,4), resized_nn.astype(np.uint8),"Resize 128×128 (nearest)")

# Row 1 — filters
show(_ax(fig,gs,1,0), mean_out,   "Mean filter k=9",  "gray")
show(_ax(fig,gs,1,1), gauss_out,  "Gaussian σ=1.5",   "gray")
show(_ax(fig,gs,1,2), median_out, "Median filter k=3 (denoised)", "gray")
show(_ax(fig,gs,1,3), sharp,      "Unsharp mask α=1.5","gray")
show(_ax(fig,gs,1,4), sobel["magnitude"], "Sobel magnitude","hot")

# Row 2 — thresholding
show(_ax(fig,gs,2,0), bin_global, "Global thresh t=128","gray")
show(_ax(fig,gs,2,1), bin_otsu,   f"Otsu  t={t:.1f}", "gray")
show(_ax(fig,gs,2,2), bin_adapt,  "Adaptive (Gaussian)","gray")
show(_ax(fig,gs,2,3), eroded,     "Morpho erode",      "gray")
show(_ax(fig,gs,2,4), dilated,    "Morpho dilate",     "gray")

# Row 3 — bit planes + histogram equalization
for b in range(4):
    show(_ax(fig,gs,3,b), cv.bit_plane_slice(img_gray,7-b),
         f"Bit plane {7-b}", "gray")
show(_ax(fig,gs,3,4), eq, "Histogram equalization","gray")

# Row 4 — transforms
show(_ax(fig,gs,4,0), img_rgb,                  "Original")
show(_ax(fig,gs,4,1), rotated.astype(np.uint8), "Rotate 30°")
show(_ax(fig,gs,4,2), cv.rotate(img_rgb,90).astype(np.uint8),  "Rotate 90°")
show(_ax(fig,gs,4,3), cv.rotate(img_rgb,180).astype(np.uint8), "Rotate 180°")
show(_ax(fig,gs,4,4), translated.astype(np.uint8), "Translate (20,20)")

# Row 5 — drawing & text
show(_ax(fig,gs,5,0), canvas,         "Drawing + Text")
canvas2 = np.zeros((H,W,3),dtype=np.uint8)
cv.draw_polygon(canvas2,
    [(128,40),(220,180),(60,180),(30,80),(226,80)],
    color=(100,200,255), filled=True)
cv.draw_rectangle(canvas2, 60,60,196,196, color=(255,200,0), thickness=3)
cv.put_text(canvas2, "POLYGON", x=60, y=210, font_scale=2, color=(255,255,255))
show(_ax(fig,gs,5,1), canvas2,        "Polygon + Rectangle")

canvas3 = img_rgb.copy()
for i in range(5):
    cv.draw_line(canvas3, x0=10+i*10, y0=10, x1=250-i*10, y1=250,
                 color=(255,50+i*40,50), thickness=2)
show(_ax(fig,gs,5,2), canvas3,        "Lines (Bresenham)")

canvas4 = np.zeros((H,W,3),dtype=np.uint8)
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
for i,(cx,cy) in enumerate([(64,64),(192,64),(128,128),(64,192),(192,192)]):
    cv.draw_point(canvas4, x=cx, y=cy, color=colors[i], radius=20)
show(_ax(fig,gs,5,3), canvas4,        "Points (filled circles)")

# Sobel Gx/Gy
gx_disp = cv.normalize(sobel["Gx"], mode="minmax")
gy_disp = cv.normalize(sobel["Gy"], mode="minmax")
show(_ax(fig,gs,5,4), gx_disp, "Sobel Gx", "RdBu_r")

# Row 6 — histograms
ax_h0 = _ax(fig,gs,6,0)
show_hist(ax_h0, counts / counts.max(), "Intensity Histogram")
ax_h1 = _ax(fig,gs,6,1)
show_hist(ax_h1, ch[:32], "Color hist R (32 bins)", "#f72585")
ax_h2 = _ax(fig,gs,6,2)
show_hist(ax_h2, ch[32:64], "Color hist G (32 bins)", "#4cc9f0")
ax_h3 = _ax(fig,gs,6,3)
show_hist(ax_h3, ch[64:], "Color hist B (32 bins)", "#7209b7")
ax_h4 = _ax(fig,gs,6,4)
show_hist(ax_h4, gh, "Gradient magnitude hist", "#f77f00")

# Row 7 — features
ax_hu = _ax(fig,gs,7,0)
show_hist(ax_hu, np.abs(hu), "Hu Moments (|log|, 7 values)", "#06d6a0")
ax_hog = _ax(fig,gs,7,1,2)
show_hist(ax_hog, hog_feat[:100], "HOG descriptor (first 100 dims)", "#ffb703")
ax_lbp = _ax(fig,gs,7,3)
show_hist(ax_lbp, lbp_feat, "LBP histogram (32 bins)", "#ef476f")

# Result summary box
ax_sum = _ax(fig,gs,7,4)
ax_sum.set_facecolor("#16213e")
ax_sum.axis("off")
color = "#06d6a0" if failed == 0 else "#ef476f"
symbol = "✓" if failed == 0 else "✗"
ax_sum.text(0.5, 0.6, f"{symbol} {passed}/{total}",
            ha="center", va="center", color=color,
            fontsize=28, fontweight="bold",
            transform=ax_sum.transAxes)
ax_sum.text(0.5, 0.3, "checks passed",
            ha="center", va="center", color="white",
            fontsize=11, transform=ax_sum.transAxes)
ax_sum.set_title("Test Results", color="white", fontsize=9, pad=4)

# ── save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "demo_output.png")
fig.savefig(out_path, dpi=130, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\n{GREEN}{BOLD}Visual output saved → {out_path}{RESET}")
print(f"Open that file to show your TA all results at once.\n")
