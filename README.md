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



# Risk Definitions

Two composite risk scores are defined:

- **RISK (v1):** Combines four factors —  
  **Tail (p95)** for high-load bursts,  
  **EMA** for recent behavior,  
  **CV** for variability, and  
  **Trend flag** for upward or downward drift.  
  Weighted as 0.5 · T + 0.3 · E + 0.1 · B + 0.1 · C.

- **RISK_v2:** A smoother version combining  
  **Mean**, **p95**, **p50**, and **EMA**  
  to balance overall level, tails, and recency.  
  Weighted as 0.20 · mean + 0.30 · p95 + 0.30 · p50 + 0.20 · EMA.

  