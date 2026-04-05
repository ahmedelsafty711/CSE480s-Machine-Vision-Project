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

### 3.3 – Padding Modes

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

---

---

## Milestone 2

---

### Feature Extraction: Colour Histogram

Separate $B$-bin histograms per channel, concatenated: dimension $3B$.

### Feature Extraction: LBP

For each pixel centre $c$ with neighbourhood sampled at $P$ equally spaced points on radius $r$:

$$\text{LBP}_{P,r}(x,y) = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p, \quad s(u) = \begin{cases}1 & u \geq 0 \\ 0 & u < 0\end{cases}$$

---

### MRMR Feature Selection

Given features $F$ and target $C$:

$$\text{MRMR} = \max_{f \in F \setminus S} \left[ I(f; C) - \frac{1}{|S|} \sum_{g \in S} I(f; g) \right]$$

where $I(\cdot; \cdot)$ is mutual information. Greedily adds one feature per iteration.

---

### KNN — Distance Metrics

| Metric     | Formula                                         |
|------------|-------------------------------------------------|
| Euclidean  | $\sqrt{\sum_i (a_i - b_i)^2}$                  |
| Manhattan  | $\sum_i |a_i - b_i|$                            |
| Cosine     | $1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ |

Euclidean computed via: $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a \cdot b^T$ (vectorised, $O(Q \cdot N \cdot D)$).

---

### Softmax Regression

**Forward:**
$$\hat{y}_k = \frac{e^{z_k - \max z}}{\sum_j e^{z_j - \max z}}, \quad z = XW + b$$

**Loss:** $\mathcal{L} = -\frac{1}{N}\sum_i \log \hat{y}_{i,c_i} + \frac{\lambda}{2}\|W\|^2$

**Gradients:**
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} X^T (\hat{Y} - Y) + \lambda W, \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_i (\hat{y}_i - y_i)$$

---

### CNN Backpropagation

**Conv2D backward:** Uses im2col transform to express convolution as matrix multiplication, then transposes to recover $dX$, $dW$.

**MaxPool backward:** Gradient flows only through the max-element (argmax mask).

**ReLU backward:** $\frac{\partial}{\partial x} \text{ReLU}(x) = \mathbf{1}[x > 0]$

---

### Optimizers

**SGD with Momentum:**
$$v_t = \beta v_{t-1} + \eta g_t, \quad \theta_t = \theta_{t-1} - v_t$$

**Adam:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$
$$\theta_t = \theta_{t-1} - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$$

---

### Evaluation Metrics

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

$$\text{Macro-F1} = \frac{1}{C}\sum_c F1_c, \quad \text{Weighted-F1} = \frac{\sum_c n_c \cdot F1_c}{\sum_c n_c}$$
