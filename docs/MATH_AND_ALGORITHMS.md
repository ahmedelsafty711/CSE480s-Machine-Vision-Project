# Math & Algorithms Notes

This document provides concise mathematical explanations and pseudocode for all major algorithms implemented in both milestones.

---

## Milestone 1

---

### 2.3 – RGB to Grayscale (ITU-R BT.601)

$$Y = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$$

The coefficients reflect the human eye's differential sensitivity to each colour channel (highest sensitivity to green).

---

### 3.1 – Image Normalization

**Min-Max Normalization:**
$$\hat{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (b - a) + a$$
Maps pixel intensities linearly from $[x_{\min}, x_{\max}]$ to $[a, b]$.

**Z-Score Normalization:**
$$\hat{x} = \frac{x - \mu}{\sigma}$$
Centres the distribution at zero with unit standard deviation.

**Fixed-Range Normalization:**
$$\hat{x} = \frac{\text{clip}(x, x_{\text{lo}}, x_{\text{hi}}) - x_{\text{lo}}}{x_{\text{hi}} - x_{\text{lo}}}$$

---

### 3.2 – Pixel Clipping

Pixel clipping is an element-wise saturation operation:

$$\hat{p}(i,j) = \min\!\bigl(\max\!\bigl(p(i,j),\; p_{\min}\bigr),\; p_{\max}\bigr)$$

Any value below $p_{\min}$ is raised to $p_{\min}$; any value above $p_{\max}$ is lowered to $p_{\max}$.  Values already inside the range are unchanged.  Implemented in one NumPy call: `np.clip(image, low, high)`.

---

| Mode       | Border behavior                                   | NumPy equivalent |
|------------|---------------------------------------------------|-----------------|
| `zero`     | Fill with 0                                       | `constant`      |
| `reflect`  | Mirror about last pixel: `... d c b | a b c d`   | `reflect`       |
| `replicate`| Repeat edge pixel: `a a a | a b c d`            | `edge`          |

---

### 3.4 – True 2-D Convolution

$$(\mathbf{f} \star \mathbf{g})[i,j] = \sum_{m} \sum_{n} f[i-m, j-n] \cdot g[m,n]$$

This differs from cross-correlation by the 180° flip of the kernel:

$$\text{Flip kernel:} \quad g'[m,n] = g[-m,-n]$$

Then apply cross-correlation with $g'$.

**Implementation using stride tricks:**
```
windows[i, j, :, :] = padded_image[i:i+kH, j:j+kW]
output = tensordot(windows, kernel_flipped, axes=[(2,3),(0,1)])
```

---

### 4.2 – Gaussian Kernel

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

In practice, normalise so the kernel sums to 1:

$$K[i, j] = e^{-\frac{(i - c)^2 + (j - c)^2}{2\sigma^2}}, \quad K \leftarrow K / \sum K$$

where $c = \lfloor k/2 \rfloor$ is the kernel centre.

---

### 4.3 – Median Filter

**Why looping is unavoidable:**  
The median is a non-linear rank-order statistic. It has no algebraic kernel representation (unlike mean or Gaussian). Therefore, we must examine the actual pixel neighbourhood values.

**Algorithm:**
```
For each pixel (i, j):
    window = all pixels in [i-r:i+r, j-r:j+r]  (flattened)
    output[i, j] = median(window)
```
Our implementation uses NumPy stride tricks to extract all windows at once as a $(H, W, k^2)$ array, then calls `np.median` along the last axis — one vectorised call per image, avoiding a Python loop over pixels.

---

### 4.4 – Thresholding

**Global threshold:**
$$T(x,y) = \begin{cases} 255 & \text{if } I(x,y) > t \\ 0 & \text{otherwise} \end{cases}$$

**Otsu's Method (maximise inter-class variance):**

Let $w_0(t)$, $w_1(t)$ be the class probabilities and $\mu_0(t)$, $\mu_1(t)$ be the class means:

$$\sigma_B^2(t) = w_0(t) \cdot w_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2$$

$$t^* = \arg\max_t \; \sigma_B^2(t)$$

Computed in $O(L)$ where $L=256$ using cumulative histograms.

**Adaptive Thresholding:**
$$T(x,y) = \text{local\_stat}(x,y) - C$$

where `local_stat` is either the local mean (box filter) or Gaussian weighted mean over a block of size $B \times B$.

---

### 4.5 – Sobel Gradients

$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

$$G_x = I \star K_x, \quad G_y = I \star K_y$$

$$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\!\left(\frac{|G_y|}{|G_x|}\right)$$

---

### 4.6 – Bit-Plane Slicing

A pixel value $p$ (uint8) is decomposed into 8 binary planes:

$$\text{plane}_b(i,j) = 255 \cdot \left(\lfloor p(i,j) / 2^b \rfloor \mod 2\right), \quad b \in \{0,\ldots,7\}$$

The MSB (b=7) plane captures the dominant structure; the LSB plane captures fine noise.

---

### 4.7 – Histogram Equalization

Given pixel CDF $\text{CDF}(v)$ and total pixels $N$:

$$\text{eq}(v) = \text{round}\!\left(\frac{\text{CDF}(v) - \text{CDF}_{\min}}{N - \text{CDF}_{\min}} \cdot 255\right)$$

Applied via a lookup table (LUT): $O(N)$ time after $O(L)$ histogram construction.

---

### 4.8 – Unsharp Masking

$$\text{mask} = I - \text{blur}(I, \sigma)$$
$$\text{out} = I + \alpha \cdot \text{mask} = (1+\alpha) I - \alpha \cdot \text{blur}(I)$$

Amplifies high-frequency detail (edges). $\alpha > 1$ gives aggressive sharpening.

---

### 4.8 – Morphological Operations

**Erosion:**
$$(\mathbf{A} \ominus \mathbf{B})[i,j] = \min_{(m,n) \in B} A[i+m, j+n]$$

**Dilation:**
$$(\mathbf{A} \oplus \mathbf{B})[i,j] = \max_{(m,n) \in B} A[i+m, j+n]$$

Both implemented with stride tricks: extract $(H, W, k^2)$ window array, then `min`/`max` along last axis.

---

### 5.1 – Resize (Backward Mapping)

For output pixel $(i, j)$, the source coordinate is:

$$\text{src}_y = i \cdot \frac{H_{\text{src}} - 1}{H_{\text{out}} - 1}, \quad \text{src}_x = j \cdot \frac{W_{\text{src}} - 1}{W_{\text{out}} - 1}$$

**Nearest-Neighbour:** $\text{src} \leftarrow \text{round}(\text{src})$

**Bilinear:**
$$f(x, y) = (1-\Delta x)(1-\Delta y) f_{00} + \Delta x(1-\Delta y) f_{10} + (1-\Delta x)\Delta y f_{01} + \Delta x \Delta y f_{11}$$

---

### 5.2 – Rotation (Backward Mapping)

For output pixel at $(x, y)$ relative to centre $(c_x, c_y)$:

$$\begin{pmatrix} x_{\text{src}} \\ y_{\text{src}} \end{pmatrix} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x - c_x \\ y - c_y \end{pmatrix} + \begin{pmatrix} c_x \\ c_y \end{pmatrix}$$

---

### 6.1b – Hu Moments

Raw moments: $m_{pq} = \sum_x \sum_y x^p y^q I(x,y)$

Central moments: $\mu_{pq} = \sum_x \sum_y (x - \bar{x})^p (y - \bar{y})^q I(x,y)$

Normalised: $\eta_{pq} = \mu_{pq} / \mu_{00}^{1+(p+q)/2}$

Seven invariants (Hu, 1962): combinations of $\eta_{pq}$ invariant to translation, scale, rotation.

Log-scaled for compactness: $\phi_i = \text{sign}(h_i) \cdot \log_{10}|h_i|$

---

### 6.2a – HOG (Histogram of Oriented Gradients)

```
1. Compute Sobel gradients (magnitude M, angle θ ∈ [0°, 180°])
2. Divide image into cells of size cell_size × cell_size
3. For each cell: build weighted orientation histogram (n_bins bins)
   vote weight = M(x,y), bin = floor(θ / (180/n_bins))
4. Group cells into overlapping blocks (block_size × block_size cells)
5. L2-normalise each block: b̂ = b / (‖b‖ + ε)
6. Concatenate all normalised blocks → feature vector
```

Output vector length:

$$L = (n_{cy} - B + 1) \cdot (n_{cx} - B + 1) \cdot B^2 \cdot n_{\text{bins}}$$

where $n_{cy} = \lfloor H / c \rfloor$, $n_{cx} = \lfloor W / c \rfloor$, $B$ = `block_size`, $c$ = `cell_size`.

---

### 6.1a – Colour Histogram

For each channel $k \in \{R, G, B\}$, divide the intensity range $[0, 255]$ into $b$ equal bins of width $w = 256 / b$:

$$\text{hist}_k[i] = \bigl|\{(x,y) \;:\; i \cdot w \leq I_k(x,y) < (i+1) \cdot w\}\bigr|$$

The final descriptor is the concatenation of all three channel histograms, optionally normalised by dividing by the total pixel count:

$$\mathbf{f} = \bigl[\text{hist}_R \;\|\; \text{hist}_G \;\|\; \text{hist}_B\bigr] \in \mathbb{R}^{3b}$$

For grayscale input only one histogram is computed, yielding $\mathbf{f} \in \mathbb{R}^b$.

---

### 6.2b – Gradient Magnitude Histogram

A global histogram of Sobel gradient magnitudes over the full image:

$$M(i,j) = \sqrt{G_x(i,j)^2 + G_y(i,j)^2}$$

$$\text{hist}[k] = \bigl|\bigl\{(i,j) \;:\; k \cdot \Delta \leq M(i,j) < (k+1) \cdot \Delta\bigr\}\bigr|$$

where $\Delta = M_{\max} / b$ and $b$ = number of bins.  Normalised to sum to 1.  This descriptor captures the global edge-energy distribution and is robust to translation and small rotations.

---

### 5.3 – Translation (Backward Mapping)

For an output pixel at position $(i, j)$, the corresponding source coordinate is obtained by inverting the shift:

$$\text{src}_y = i - t_y, \quad \text{src}_x = j - t_x$$

If $(src_y, src_x)$ falls outside the image boundary $[0, H-1] \times [0, W-1]$, the output pixel is assigned the fill value (default 0).  Only positions inside the boundary are sampled (bilinear by default).

$$\text{out}[i,j] = \begin{cases} I[i - t_y,\; j - t_x] & \text{if } 0 \leq i-t_y \leq H-1 \text{ and } 0 \leq j-t_x \leq W-1 \\ v_{\text{fill}} & \text{otherwise} \end{cases}$$

---

### Section 7 – Drawing Primitives

#### Bresenham's Line Algorithm

Rasterises a straight line between integer endpoints $(x_0, y_0)$ and $(x_1, y_1)$ using only integer arithmetic.  Let $dx = |x_1 - x_0|$, $dy = |y_1 - y_0|$, $s_x = \text{sign}(x_1 - x_0)$, $s_y = \text{sign}(y_1 - y_0)$:

```
err = dx - dy
loop:
    plot(x, y)
    if x == x1 and y == y1: break
    e2 = 2 * err
    if e2 > -dy: err -= dy;  x += sx
    if e2 <  dx: err += dx;  y += sy
```

The error term `err` tracks the accumulated fractional deviation from the ideal line; when it exceeds the half-step threshold the algorithm steps in the minor axis direction.  Time complexity: $O(\max(|dx|, |dy|))$.

#### Filled Polygon – Scanline Rasterisation

For each horizontal scanline $y$ between $y_{\min}$ and $y_{\max}$:

1. Find all edge intersections with row $y$:

$$x_{\text{int}} = x_a + \frac{(y - y_a)(x_b - x_a)}{y_b - y_a} \quad \text{for each edge } (a,b) \text{ where } y_a \neq y_b$$

2. Sort intersection x-coordinates.
3. Fill between each consecutive pair: $[x_{\text{int}}_{2k},\; x_{\text{int}}_{2k+1}]$.

#### Circle / Point

A filled circle of radius $r$ centred at $(x_c, y_c)$ is rasterised by testing every pixel $(x_c + dx, y_c + dy)$ for $dx, dy \in [-(r-1), r-1]$:

$$\text{fill if} \quad dx^2 + dy^2 < r^2$$

---
