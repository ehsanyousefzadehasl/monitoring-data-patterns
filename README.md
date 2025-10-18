# Monitoring Patterns and Statistical Metrics

Monitoring GPU hardware counters (such as **SMACT**, **SMOCC**, and **DRAMA** for NVIDIA GPUs) over sliding windows reveals characteristic **patterns** of activity.  
Recognizing these patterns helps assess *collocation opportunities* — whether placing another workload on the same GPU will cause dramatically harmful interference and consequnetly drastic slowdowns.

---

## Possible GPU Utilization Patterns

We synthetically categorize the potential patterns in monitored data and look into their statistical measurements. These patterns can indicate how risky a GPU is for workload collocation or adding more load on it.

## GPU Utilization Patterns

Monitoring GPU hardware counters such as **SMACT**, **SMOCC**, and **DRAMA** over sliding windows reveals characteristic **patterns of activity**.  

| Pattern | Description | Collocation Implication |
|----------|--------------|--------------------------|
| **Idle / Flat Low** | Values near zero with very low variance. | **Safe** — GPU mostly unused, large headroom for new tasks. |
| **Steady Busy** | Values consistently high with low variance. | **Unsafe** — GPU fully engaged, no available capacity and collocation can result in dramatic slowdowns to the collocated tasks. |
| **Bursty / Spiky** | Mostly low-to-moderate activity with intermittent peaks. Percentiles (p95, p99) much higher than mean. | Overally collocation can be beneficial if the the spikey part is short-living. |
| **Trending Up / Down** | Gradual increase or decrease in utilization within the window. | It can be both beneficial or harmful regarding the ranges it is changing, it happens when a task warms up or has a repeating like that! |
| **Bimodal / On-Off Switching** | Alternating periods of high and low utilization. | **Unstable** — interference risk depends on phase overlap if there is a gurantee for that, which in high level task collocation using MPS can be beneficial. |
| **Spike then Cool-off or vice versa** | Short-lived peak followed by stable low utilization. Percentiles capture the spike; EMA highlights the stable end. | **Safe after cool-off** — only once GPU stabilizes. Again if the spike is short living, offers a lot of room for performance gains.|


---

**Notes:**  
- *SMACT* and *SMOCC* capture compute intensity and warp activity.  
- *DRAMA* reflects DRAM bandwidth saturation.


## Statistical Metrics

For each GPU metric stream (**SMACT**, **SMOCC**, **DRAMA**), we compute a set of descriptive statistics to characterize utilization behavior within a window.

---

### **1. Mean**
**Formula:** `mean(x) = (1/N) * Σ xᵢ`  
- **Reveals:** overall average utilization  
- **Good for:** steady, stable patterns  
- **Weakness:** hides short bursts

---

### **2. Median**
**Formula:** `median(x) = p₅₀(x)`  
- **Reveals:** central tendency robust to outliers  
- **Good for:** noisy data with occasional spikes  

---

### **3. Percentiles (p95, p99)**
**Formula:** `p_q(x) = smallest v such that (# of xᵢ ≤ v)/N ≥ q`  
- **Reveals:** tail behavior (bursts or spikes)  
- **Good for:** identifying rare but impactful utilization peaks  
- **Weakness:** ignores sustained load if only a few samples are high  

---

### **4. Exponential Moving Average (EMA)**
**Formula:** `EMAₜ = α·xₜ + (1−α)·EMAₜ₋₁` , with smoothing factor `α ∈ (0,1)`  
- **Reveals:** recent trend; emphasizes latest samples  
- **Good for:** quickly detecting shifts or recent changes  
- **Weakness:** sensitive to the choice of `α`  

---

### **5. Coefficient of Variation (CV)**
**Formula:** `CV(x) = std(x) / mean(x)`  
- **Reveals:** relative burstiness (variance normalized by mean)  
- **Good for:** identifying unstable or highly variable workloads  
- **Weakness:** unreliable when mean ≈ 0  

---

### **6. Median Absolute Deviation (MAD)**
**Formula:** `MAD(x) = median(|xᵢ − median(x)|)`  
- **Reveals:** robust spread of data around the median  
- **Good for:** measuring variability without being skewed by outliers  

---

Together, these metrics provide a **composite view** of GPU activity — enabling both:  
- **Pattern categorization** (e.g., *Idle*, *Steady*, *Bursty*), and  
- **Quantitative scoring** (e.g., a *collocation risk metric*).  



## Per-Metric Trend Flags and Composite Risk

We compute **all statistics per metric stream** — separately for **SMACT**, **SMOCC**, and **DRAMA** — over the window. Let the three series be x_S (SMACT), x_O (SMOCC), x_D (DRAMA).

### Per-Metric Features (computed for each of: SMACT, SMOCC, DRAMA)
- mean(x)          : overall average
- median(x)        : robust central tendency
- p95(x), p99(x)   : tail (bursts/spikes)
- EMA_last(x)      : exponential moving average at window end
- CV(x)            : coefficient of variation = std(x) / mean(x)
- MAD(x)           : median absolute deviation
- slope(x)         : linear-regression slope of x vs. time (least squares)
- trend_flag(x)    : 1 if |slope(x)| > τ, else 0  (τ is a small threshold)

**Notes**
- EMA_last uses α either manually set or α ≈ 2/(N+1) when auto-derived.
- trend_flag is per metric; a series with noticeable drift (up/down) within the window sets its own flag to 1.

### Aggregating Per-Metric Features into a Single Per-GPU Score
Define the **per-metric** components:
- Tail per metric:      T_S = p95(x_S), T_O = p95(x_O), T_D = p95(x_D)
- Recency per metric:   E_S = EMA_last(x_S), E_O = EMA_last(x_O), E_D = EMA_last(x_D)
- Burstiness per metric: B_S = CV(x_S), B_O = CV(x_O), B_D = CV(x_D)
- Trend per metric:      C_S = trend_flag(x_S), C_O = trend_flag(x_O), C_D = trend_flag(x_D)

Combine to **per-GPU components** (default is a conservative “max” across metrics):
- T = max(T_S, T_O, T_D)
- E = max(E_S, E_O, E_D)
- B = max(B_S, B_O, B_D)
- C = 1 if any of {C_S, C_O, C_D} is 1, else 0

(Alternative: use a weighted sum across metrics when you want to favor/penalize specific resources.)

### Composite Risk (per GPU, per window)
RISK = wT*T + wE*E + wB*B + wC*C  
Default weights: wT=0.5, wE=0.3, wB=0.1, wC=0.1 (tunable in YAML).

**Interpretation**
- T (tail) dominates: large p95 in any metric signals bursty high load.
- E (EMA) adds recency: what the GPU “feels like now.”
- B (CV) penalizes instability even if averages look OK.
- C (trend) penalizes windows with clear upward/downward drift.

**Hysteresis for Decisions**

