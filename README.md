# Monitoring Patterns and Statistical Metrics

Monitoring GPU hardware counters (such as **SMACT**, **SMOCC**, and **DRAMA**) over sliding windows reveals characteristic **patterns** of activity.  
Recognizing these patterns helps assess *collocation risk* — whether placing another workload on the same GPU will cause dramatically harmful interference.

---

## Patterns of GPU Utilization

1. **Idle / Flat Low**  
   - Values near zero, very low variance.  
   - GPU mostly unused; safe for collocation.

2. **Steady Busy**  
   - Values consistently high with low variance.  
   - GPU fully engaged; no headroom for new tasks.

3. **Bursty / Spiky**  
   - Mostly low-to-moderate activity with intermittent peaks.  
   - Average hides short spikes; percentiles capture them.  
   - Risky: bursts may overlap with other workloads.

4. **Trending Up / Trending Down**  
   - Gradual increase or decrease in utilization within the window.  
   - Indicates load is shifting; scheduler should be conservative.  

5. **Bimodal / On-Off Switching**  
   - Alternating high and low states.  
   - Leads to instability; risk depends on overlap of cycles.  

6. **Memory-Pressure Dominated**  
   - DRAMA (DRAM Active) is persistently high even if SMACT/SMOCC are moderate.  
   - Memory bandwidth bottleneck; collocating another memory-heavy task causes interference.  

7. **Spike then Cool-off**  
   - Short-lived peak followed by stable low utilization.  
   - Percentiles highlight the spike, EMA highlights the stable end.  
   - Safe for collocation only after cool-off.

---
## Statistical Metrics

For each metric stream (**SMACT**, **SMOCC**, **DRAMA**) we compute the following:

### 1. Mean
Formula: mean(x) = (1/N) * Σ x_i  
- Reveals: overall average utilization.  
- Good for: steady patterns.  
- Weakness: hides bursts.

### 2. Median
Formula: median(x) = 50th percentile of x  
- Reveals: central tendency robust to outliers.  
- Good for: noisy data with occasional spikes.

### 3. Percentiles (p95, p99)
Formula: p_q(x) = smallest v such that (# of x_i ≤ v) / N ≥ q  
- Reveals: tail behavior (bursts, spikes).  
- Good for: detecting rare but impactful high utilization.  
- Weakness: insensitive to sustained high load if only a few points.

### 4. Exponential Moving Average (EMA)
Formula: EMA_t = α * x_t + (1−α) * EMA_(t−1), with smoothing factor α ∈ (0,1)  
- Reveals: recent trend, weights latest samples more.  
- Good for: catching shifts quickly.  
- Weakness: sensitive to α.

### 5. Coefficient of Variation (CV)
Formula: CV(x) = std(x) / mean(x)  
- Reveals: relative burstiness (variance scaled by mean).  
- Good for: identifying unstable or spiky workloads.  
- Weakness: unstable if mean ≈ 0.

### 6. Median Absolute Deviation (MAD)
Formula: MAD(x) = median(|x_i − median(x)|)  
- Reveals: robust spread of data around the median.  
- Good for: detecting variability without being skewed by outliers.


Together, these metrics provide a **composite view** of GPU behavior.  
This allows both:
- **Pattern categorization** (Idle, Steady, Bursty, etc.), and  
- **Quantitative scoring** (e.g., a collocation risk metric).  

---


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

