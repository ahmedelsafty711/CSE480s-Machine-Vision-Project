# Math & Algorithms Notes

Concise mathematical explanations and pseudocode for every algorithm implemented across Milestone 1 and Milestone 2.

---

# Milestone 1

---

## 2.3 — RGB to Grayscale (ITU-R BT.601)

$$Y = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$$

The coefficients reflect the human eye's differential sensitivity to each colour channel. Green contributes the most (~59%) because human photoreceptors are most sensitive in the green-yellow range. Output is float32 in [0, 1].

---

## 3.1 — Image Normalization (3 Modes)

**Min-Max:**
$$\hat{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (b - a) + a$$

Maps pixel intensities linearly from $[x_{\min}, x_{\max}]$ to $[a, b]$. Edge case: if $x_{\min} = x_{\max}$ (constant image), output is all $a$.

**Z-Score:**
$$\hat{x} = \frac{x - \mu}{\sigma}$$

Centres the distribution at zero with unit standard deviation. If $\sigma = 0$, output is all zeros.

**Fixed-Range:**
$$\hat{x} = \frac{\text{clip}(x,\, x_{\text{lo}},\, x_{\text{hi}}) - x_{\text{lo}}}{x_{\text{hi}} - x_{\text{lo}}}$$

Clips first, then rescales to [0, 1].

---

## 3.3 — Padding Modes

| Mode | Border Behaviour | NumPy Equivalent |
|------|-----------------|------------------|
| `zero` | Fill with 0 | `constant` |
| `reflect` | Mirror about last pixel: `d c b | a b c d` | `reflect` |
| `replicate` | Repeat edge pixel: `a a a | a b c d` | `edge` |

---

## 3.4 — True 2-D Convolution

$$(\mathbf{f} \star \mathbf{g})[i,j] = \sum_{m}\sum_{n} f[i-m,\, j-n] \cdot g[m,n]$$

Differs from cross-correlation by the 180° flip of the kernel:

$$g'[m,n] = g[-m,-n]$$

**Implementation via stride tricks:**
```
1. Flip kernel 180°: k_flip = kernel[::-1, ::-1]
2. Pad image by (kH//2, kW//2) on all sides
3. Extract sliding windows: windows[i,j,:,:] = padded[i:i+kH, j:j+kW]
   Shape: (H, W, kH, kW)  — via np.lib.stride_tricks.as_strided
4. output = tensordot(windows, k_flip, axes=[(2,3),(0,1)])
```

No Python loops over pixels. O(H·W·kH·kW) with fully vectorized multiply-accumulate.

---

## 4.2 — Gaussian Kernel

$$G(x, y) = \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right), \quad K \leftarrow \frac{K}{\sum K}$$

Normalise so the kernel sums to 1 (energy-preserving convolution).

---

## 4.3 — Median Filter

The median is a non-linear rank-order statistic with no algebraic kernel representation.

**Why a loop is unavoidable:** `median(a·x + b·y) ≠ a·median(x) + b·median(y)`. The operation cannot be expressed as a convolution.

**Vectorised implementation:**
```
1. Pad image by (ksize//2) on all sides (replicate)
2. Extract all (H, W, ksize²) windows via stride tricks
3. np.median(windows.reshape(H, W, -1), axis=2)
   → one vectorised call, no Python loop over pixels
```

---

## 4.4 — Thresholding

**Global:**
$$T(x,y) = \begin{cases} 255 & \text{if } I(x,y) > t \\ 0 & \text{otherwise} \end{cases}$$

**Otsu's Method — maximise between-class variance:**

Let $w_0(t)$, $w_1(t)$ be class probabilities and $\mu_0(t)$, $\mu_1(t)$ class means:

$$\sigma_B^2(t) = w_0(t)\cdot w_1(t)\cdot[\mu_0(t) - \mu_1(t)]^2$$

$$t^* = \arg\max_t \; \sigma_B^2(t)$$

Computed in $O(L)$ where $L=256$ using cumulative histogram sums.

**Adaptive:**
$$T(x,y) = \text{local\_stat}(x,y) - C$$

where `local_stat` is either the local mean (box filter) or Gaussian weighted mean over block $B \times B$.

---

## 4.5 — Sobel Gradients

$$K_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}, \quad K_y = \begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$$

$$G_x = I \star K_x, \quad G_y = I \star K_y$$

$$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\!\left(\frac{|G_y|}{|G_x|}\right) \in [0°, 180°]$$

---

## 4.6 — Bit-Plane Slicing

$$\text{plane}_b(i,j) = 255 \cdot \left(\lfloor p(i,j) / 2^b \rfloor \bmod 2\right), \quad b \in \{0,\ldots,7\}$$

Equivalent to: `((pixel >> b) & 1) * 255`. The MSB plane ($b=7$) captures dominant structure; the LSB plane ($b=0$) captures fine noise.

---

## 4.7 — Histogram Equalization

Given pixel CDF and total pixels $N$:

$$\text{eq}(v) = \text{round}\!\left(\frac{\text{CDF}(v) - \text{CDF}_{\min}}{N - \text{CDF}_{\min}} \cdot 255\right)$$

Applied via a lookup table: $O(N)$ time after $O(L)$ histogram construction.

---

## 4.8 — Unsharp Masking

$$\text{mask} = I - \text{blur}(I,\,\sigma)$$
$$\text{out} = I + \alpha \cdot \text{mask} = (1+\alpha)\,I - \alpha \cdot \text{blur}(I)$$

Amplifies high-frequency detail (edges). $\alpha > 1$ gives aggressive sharpening.

---

## 4.8 — Morphological Operations

**Erosion:**
$$(\mathbf{A} \ominus \mathbf{B})[i,j] = \min_{(m,n) \in B} A[i+m,\, j+n]$$

**Dilation:**
$$(\mathbf{A} \oplus \mathbf{B})[i,j] = \max_{(m,n) \in B} A[i+m,\, j+n]$$

Both use stride tricks: extract $(H, W, kH \cdot kW)$ window array, then `min`/`max` along last axis.

---

## 5.1 — Resize (Backward Mapping)

For output pixel $(i, j)$:

$$\text{src}_y = i \cdot \frac{H_{\text{src}} - 1}{H_{\text{out}} - 1}, \quad \text{src}_x = j \cdot \frac{W_{\text{src}} - 1}{W_{\text{out}} - 1}$$

**Nearest-Neighbour:** $\text{src} \leftarrow \text{round}(\text{src})$

**Bilinear:**
$$f(x,y) = (1\!-\!\Delta x)(1\!-\!\Delta y)\,f_{00} + \Delta x(1\!-\!\Delta y)\,f_{10} + (1\!-\!\Delta x)\Delta y\,f_{01} + \Delta x\,\Delta y\,f_{11}$$

---

## 5.2 — Rotation (Backward Mapping)

For output pixel at $(x,y)$ relative to centre $(c_x, c_y)$:

$$\begin{pmatrix}x_{\text{src}}\\y_{\text{src}}\end{pmatrix} = \begin{pmatrix}\cos\theta & \sin\theta\\-\sin\theta & \cos\theta\end{pmatrix}\begin{pmatrix}x - c_x\\y - c_y\end{pmatrix} + \begin{pmatrix}c_x\\c_y\end{pmatrix}$$

Pixels that map outside the source boundary are filled with `fill_value`.

---

## 5.3 — Translation (Backward Mapping)

$$\text{src}_y = \text{out}_y - t_y, \quad \text{src}_x = \text{out}_x - t_x$$

Pixels mapping outside the input boundary are filled with `fill_value`.

---

## 6.1b — Hu Moments

Raw moments: $m_{pq} = \sum_x \sum_y x^p y^q I(x,y)$

Central moments: $\mu_{pq} = \sum_x \sum_y (x-\bar{x})^p (y-\bar{y})^q I(x,y)$

Normalised: $\eta_{pq} = \mu_{pq} / \mu_{00}^{1+(p+q)/2}$

Seven Hu invariants (1962): combinations of $\eta_{pq}$ invariant to translation, scale, rotation.

Log-scaled for compactness: $\phi_i = \text{sign}(h_i) \cdot \log_{10}|h_i|$

---

## 6.2a — HOG (Histogram of Oriented Gradients)

```
1. Compute Sobel gradients: magnitude M, angle θ ∈ [0°,180°]
2. Divide image into cells of size cell_size × cell_size
3. For each cell: build orientation histogram (n_bins bins)
      vote weight = M(x,y),  bin = floor(θ / (180/n_bins))
4. Group cells into overlapping blocks (block_size × block_size cells)
5. L2-normalise each block: b̂ = b / (‖b‖ + ε)
6. Concatenate all normalised blocks → feature vector
```

---

## 7 — Bresenham's Line Algorithm

```
dx = |x1-x0|,  dy = |y1-y0|
sx = sign(x1-x0),  sy = sign(y1-y0)
err = dx - dy

while True:
    plot(x0, y0)
    if x0 == x1 and y0 == y1: break
    e2 = 2 * err
    if e2 > -dy:  err -= dy;  x0 += sx
    if e2 <  dx:  err += dx;  y0 += sy
```

Iterates over the dominant axis, tracking an integer error term to decide when to step on the minor axis. O(max(|Δx|, |Δy|)) with integer arithmetic only.

---

## 7 — Scanline Polygon Fill

```
For each scanline y from y_min to y_max:
    Find all x-intersections of edges with row y:
        For edge (ax,ay)→(bx,by) where min(ay,by) ≤ y < max(ay,by):
            x_int = ax + (y - ay) * (bx - ax) / (by - ay)
    Sort x-intersections
    Fill between pairs: image[y, x_pairs[0]:x_pairs[1]], ...
```

The even-odd rule: fill between the 1st and 2nd intersection, 3rd and 4th, etc.

---

---

# Milestone 2

---

## 1 — Dataset Generation

Six synthetic classes of geometric shapes are drawn onto random background canvases using the minicv drawing library. Intra-class variability is achieved by:

- Random shape position (uniform over canvas with margin)
- Random shape size (uniform within class-specific range)
- Random shape rotation (uniform over [0°, 360°])
- Random shape colour (random HSV → RGB)
- Random background colour + optional linear gradient
- Random brightness scaling (simulates lighting variability)
- Additive Gaussian noise, σ sampled per image from [5, 18]

**Stratified split:** 70% train / 15% val / 15% test. Stratified means each class is split independently, guaranteeing equal class representation in every partition.

---

## 2 — Preprocessing

**Resize to 64×64:** all images are brought to a canonical shape via bilinear backward mapping (minicv.resize). Fixed size is required because feature vectors and model inputs must all have the same dimensionality.

**Normalization — choice justification:**

For feature-based models (KNN, Softmax): min-max to [0, 1].
$$\hat{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

For neural models (CNN, MobileNetV3): per-channel z-score.
$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c + \varepsilon}$$

Z-score gives zero-mean, unit-variance inputs per channel. This stabilizes gradient magnitudes at initialization — no single channel dominates the first layer's gradient just because its pixel values are numerically larger.

---

## 3 — Augmentation

All 6 transforms use the minicv library as the backend. Augmentation is applied only to training images.

| Transform | Implementation | Invariance Taught |
|-----------|---------------|------------------|
| Horizontal flip | `image[:, ::-1, :]` | Mirror symmetry |
| Random rotate ±20° | `minicv.rotate()` backward mapping | Rotation |
| Random translate ±8px | `minicv.translate()` backward mapping | Position |
| Brightness jitter ×[0.6, 1.4] | pixel-wise scalar multiply | Lighting |
| Gaussian noise N(0, σ) | NumPy random + clip | Sensor noise |
| Random crop + resize | slice + `minicv.resize()` | Scale / viewpoint |

**Per-epoch augmentation:** augmentation is re-applied with fresh random parameters every epoch, so the same base image produces a different augmented version each time. This effectively multiplies dataset size by the number of training epochs.

---

## 4.1 — Feature Pool

Four feature families are extracted per image and concatenated into a 199-dimensional vector.

### Color Histogram (96-d)

Per-channel histogram over 32 intensity bins, normalized by pixel count:

$$h_c[k] = \frac{1}{N} \sum_{i,j} \mathbf{1}\!\left[\frac{255k}{B} \leq I_c(i,j) < \frac{255(k+1)}{B}\right]$$

for channel $c \in \{R, G, B\}$, bin $k \in \{0,\ldots,B-1\}$, $B=32$.

### LBP — Local Binary Pattern (32-d)

For each pixel $(i,j)$ with center value $g_c$, sample $P=8$ neighbours on a circle of radius $R=1$:

$$\text{LBP}(i,j) = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p, \quad s(x) = \begin{cases}1 & x \geq 0 \\ 0 & x < 0\end{cases}$$

Neighbour positions are bilinearly interpolated (sub-pixel). The LBP codes are histogrammed into 32 bins. LBP is invariant to monotonic intensity transformations.

### Gradient Magnitude Histogram (64-d)

Apply Sobel, compute $|G| = \sqrt{G_x^2 + G_y^2}$, histogram magnitudes into 64 bins normalized by total count. Captures global edge energy distribution.

### Hu Moments (7-d)

See Milestone 1 Section 6.1b. Log-scaled invariants capture global shape properties independent of position, scale, and rotation.

### Feature Index Scheme

| Family | Dim | Indices |
|--------|-----|---------|
| color_histogram | 96 | [0:96] |
| lbp | 32 | [96:128] |
| gradient_hist | 64 | [128:192] |
| hu_moments | 7 | [192:199] |
| **Total** | **199** | |

---

## 4.2 — MRMR Feature Selection

**Minimum Redundancy Maximum Relevance** (Ding & Peng, 2005).

### Mutual Information

$$I(A;B) = H(A) + H(B) - H(A,B) = \sum_{a,b} p(a,b) \log_2\frac{p(a,b)}{p(a)\,p(b)}$$

Estimated via 2-D joint histogram with $B=10$ bins per axis.

### Greedy Forward Selection

```
S = {}   (selected feature set)
remaining = {0, 1, ..., D-1}

Step 1:  f1 = argmax_{f} I(f; y)
         S ← {f1},  remaining ← remaining \ {f1}

For step t = 2, ..., K:
    For each candidate f in remaining:
        score(f) = I(f; y)  −  (1/|S|) · Σ_{s∈S} I(f; s)
                   ↑ relevance      ↑ redundancy penalty
    ft = argmax_{f} score(f)
    S ← S ∪ {ft},  remaining ← remaining \ {ft}

Return S
```

The algorithm is $O(K \cdot D \cdot N)$ in MI evaluations. Fit on training data only; the selected index array is applied identically to val and test.

---

## 5.1 — KNN

### Vectorised Euclidean Distance

For test matrix $X_{\text{te}} \in \mathbb{R}^{M \times D}$ and training matrix $X_{\text{tr}} \in \mathbb{R}^{N \times D}$:

$$\|a - b\|^2 = \|a\|^2 - 2\,a^\top b + \|b\|^2$$

```
sq_te  = sum(X_te²,  axis=1, keepdims=True)   # (M, 1)
sq_tr  = sum(X_tr²,  axis=1, keepdims=True).T  # (1, N)
cross  = X_te @ X_tr.T                          # (M, N)
D      = sqrt(max(sq_te - 2*cross + sq_tr, 0)) # (M, N)
```

No Python loop over test examples. Complexity: $O(MND)$ with BLAS matrix multiply.

### Inference

```
For each test example i:
    nn_idx = argsort(D[i])[:k]
    votes  = bincount(y_train[nn_idx])
    pred[i] = argmax(votes)
```

### k-Sweep

Evaluate on validation set at $k \in \{1,3,5,7,9,11,15\}$. Report best $k$ and its validation accuracy.

---

## 5.2 — Softmax Regression

### Forward Pass

$$z = XW + b \quad (N \times C), \qquad p_i = \frac{\exp(z_i - \max z)}{\sum_j \exp(z_j - \max z)}$$

Subtract max for numerical stability (prevents overflow in exp).

### Cross-Entropy Loss

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log\!\left(\max(p_i[y_i],\, \varepsilon)\right), \quad \varepsilon = 10^{-7}$$

Epsilon clipping prevents $\log(0) = -\infty$.

### Gradient (Elegant Closed Form)

$$\frac{\partial L}{\partial z} = \frac{1}{N}(P - Y_{\text{one-hot}})$$

$$\frac{\partial L}{\partial W} = X^\top \cdot \frac{\partial L}{\partial z}, \qquad \frac{\partial L}{\partial b} = \sum_i \frac{\partial L}{\partial z_i}$$

### Mini-Batch Training Loop

```
For each epoch:
    idx = shuffle(0..N-1)
    For each batch b of size B from idx:
        p     = softmax(X_b @ W + b_bias)
        loss  = cross_entropy(p, y_b)
        dW    = X_b.T @ (p - one_hot(y_b)) / B
        db    = mean(p - one_hot(y_b), axis=0)
        dW   += λ · W                   (L2 regularization)
        clip_gradients(dW, db, max_norm)
        optimizer.step(W, b_bias, dW, db)
    early_stopping.update(val_loss)
```

---

## 5.3 — CNN from Scratch

### im2col: Convolution as Matrix Multiply

**Forward:**
```
X_col = im2col(X_pad, kH, kW, H_out, W_out)
         Shape: (C_in·kH·kW,  N·H_out·W_out)
W_2d  = W.reshape(F, -1)
         Shape: (F,  C_in·kH·kW)
Y_2d  = W_2d @ X_col + b
         Shape: (F,  N·H_out·W_out)
Y     = Y_2d.reshape(F, N, H_out, W_out).transpose(1,0,2,3)
```

**Backward:**
```
dY_2d = dY.transpose(1,0,2,3).reshape(F, -1)
dW_2d = dY_2d @ X_col.T                # filter gradient
dX_col = W_2d.T @ dY_2d               # input gradient
dX_pad = col2im(dX_col, ...)           # scatter back
dX    = dX_pad[:, :, p:-p, p:-p]      # remove padding
```

### ReLU

$$f(x) = \max(0, x), \qquad \frac{\partial f}{\partial x} = \mathbf{1}[x > 0]$$

Forward: zero out negatives. Backward: pass gradient where $x > 0$, zero elsewhere.

### Max Pooling

$$y[i,j] = \max_{(m,n) \in W_{ij}} x[m,n]$$

**Forward:** cache the argmax positions.

**Backward:** route gradient only to the position that held the maximum:

$$\frac{\partial L}{\partial x[m,n]} = \begin{cases}\frac{\partial L}{\partial y[i,j]} & \text{if }(m,n) = \arg\max W_{ij} \\ 0 & \text{otherwise}\end{cases}$$

### Flatten & FC

Flatten: reshape $(N,C,H,W) \to (N, C \cdot H \cdot W)$. Backward: reshape gradient back.

FC forward: $y = XW + b$. Backward: $dW = X^\top dY$, $dX = dY\,W^\top$.

### Full Architecture

```
Input (N, 3, 32, 32)
→ Conv(3→16, 3×3, pad=1)  + ReLU + MaxPool(2)   → (N, 16, 16, 16)
→ Conv(16→32, 3×3, pad=1) + ReLU + MaxPool(2)   → (N, 32,  8,  8)
→ Flatten                                          → (N, 2048)
→ FC(2048→128) + ReLU
→ FC(128→6)
→ SoftmaxCELoss
```

---

## 5.4 — MobileNetV3-Small (Howard et al., ICCV 2019)

### Depthwise Separable Convolution

Standard conv: $C_{\text{in}} \cdot kH \cdot kW \cdot C_{\text{out}}$ multiplications per position.

Factorised:
1. **Depthwise conv:** one filter per input channel → $C_{\text{in}} \cdot kH \cdot kW$
2. **Pointwise conv (1×1):** mix channels → $C_{\text{in}} \cdot C_{\text{out}}$

Total: $C_{\text{in}}(kH \cdot kW + C_{\text{out}})$ vs. $C_{\text{in}} \cdot kH \cdot kW \cdot C_{\text{out}}$. For $k=3$, $C_{\text{out}}=32$: ~$8\times$ fewer operations.

### Inverted Residual Block

```
Input (C_in channels)
→ 1×1 pointwise expand  (C_in → C_exp)
→ 3×3 or 5×5 depthwise  (groups=C_exp)
→ SE block (optional)
→ 1×1 pointwise project (C_exp → C_out)
→ skip connection if stride=1 and C_in == C_out
```

### Squeeze-and-Excitation

$$s = \text{GlobalAvgPool}(x) \quad (C,)$$
$$w = \text{HardSigmoid}(\text{FC}_2(\text{ReLU}(\text{FC}_1(s)))) \quad (C,)$$
$$\text{out} = x \cdot w$$

Learns per-channel attention weights, amplifying informative channels.

### Hard-Swish Activation

Approximation of $\text{swish}(x) = x \cdot \sigma(x)$:

$$\text{hard-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

where $\text{ReLU6}(x) = \min(\max(0,x), 6)$. Matches swish in $[-3, 3]$, saturates outside. Integer-arithmetic-friendly.

---

## 6 — Optimizers

### SGD

$$W \leftarrow W - \eta \cdot \nabla W$$

### Adam

Maintains first and second moment estimates per parameter:

$$m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1)\,g_t$$
$$v_t \leftarrow \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$$

Bias-corrected estimates:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

Update:
$$W \leftarrow W - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Default: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$.

### Learning Rate Schedules

**Step Decay:**
$$\eta(e) = \eta_0 \cdot \gamma^{\lfloor e / N \rfloor}$$

**Exponential Decay:**
$$\eta(e) = \eta_0 \cdot \gamma^e$$

**Reduce on Plateau:**
```
if val_loss hasn't improved for patience epochs:
    η ← max(η × factor, min_lr)
```

### Safety Features

**Early Stopping:**
```
best_loss = ∞,  counter = 0
For each epoch:
    if val_loss < best_loss − δ:
        best_loss = val_loss,  counter = 0
    else:
        counter += 1
        if counter ≥ patience: stop training
```

**Gradient Clipping:**
$$\|g\|_2 = \sqrt{\sum_{W} \sum_{i,j} (dW_{ij})^2}$$
$$\text{if } \|g\|_2 > g_{\max}: \quad g \leftarrow g \cdot \frac{g_{\max}}{\|g\|_2}$$

**L2 Regularization:**
$$L_{\text{reg}} = L + \frac{\lambda}{2}\|W\|^2, \qquad \frac{\partial L_{\text{reg}}}{\partial W} = \frac{\partial L}{\partial W} + \lambda W$$

Applied by adding $\lambda W$ to $dW$ before the optimizer step.

---

## 8 — Evaluation Metrics (From Scratch)

### Confusion Matrix

$$\text{CM}[i,j] = \#\{\text{examples with true label } i \text{ predicted as } j\}$$

Diagonal = correct; off-diagonal = specific class confusions.

### Per-Class Metrics from CM

For class $c$:
$$\text{TP}_c = \text{CM}[c,c], \quad \text{FP}_c = \sum_i \text{CM}[i,c] - \text{TP}_c, \quad \text{FN}_c = \sum_j \text{CM}[c,j] - \text{TP}_c$$

$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \varepsilon}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c + \varepsilon}$$

$$F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c + \varepsilon}$$

### Macro-F1 and Weighted-F1

$$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^C F1_c$$

$$\text{Weighted-F1} = \frac{\sum_{c=1}^C \text{support}_c \cdot F1_c}{\sum_{c=1}^C \text{support}_c}$$

where $\text{support}_c = \sum_j \text{CM}[c,j]$ is the number of true examples for class $c$.

Macro-F1 weights all classes equally (use when all classes matter equally). Weighted-F1 weights by class size (use with class imbalance, matches overall accuracy more closely).

---

## Initialisation Strategies

| Layer | Initialisation | Rationale |
|-------|---------------|-----------|
| FC layers | He: $W \sim \mathcal{N}(0,\, \sqrt{2/\text{fan\_in}})$ | Preserves variance through ReLU |
| Conv filters | He: $W \sim \mathcal{N}(0,\, \sqrt{2/(C_{\text{in}} \cdot kH \cdot kW)})$ | Prevents vanishing/exploding activations |
| Softmax W | Glorot: $W \sim \mathcal{N}(0,\, \sqrt{2/(\text{fan\_in}+\text{fan\_out})})$ | No ReLU, symmetric |
| Biases | $b = 0$ | Standard |

---

## Weight Initialization: Why It Matters

If weights are initialized too large, activations explode through layers. If too small, they vanish. He initialization maintains activation variance at approximately 1 for ReLU networks:

$$\text{Var}(y) = \text{Var}(Wx) = n \cdot \text{Var}(W) \cdot \text{Var}(x)$$

Setting $\text{Var}(W) = 2/n$ (where $n$ = fan-in, the $2$ accounts for ReLU zeroing half the activations) keeps $\text{Var}(y) \approx \text{Var}(x)$ through all layers.
