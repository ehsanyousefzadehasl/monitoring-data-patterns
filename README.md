# Monitoring Patterns and Statistical Metrics

Monitoring GPU hardware counters (such as **SMACT**, **SMOCC**, and **DRAMA**) over sliding windows reveals characteristic **patterns** of activity.  
Recognizing these patterns helps assess *collocation risk* — whether placing another workload on the same GPU will cause interference.

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
