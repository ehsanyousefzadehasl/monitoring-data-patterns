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

For each metric stream (**SMACT**, **SMOCC**, **DRAMA**), we compute window statistics.  
These provide complementary perspectives on GPU behavior.

### 1. Mean
\[
\text{mean}(x) = \frac{1}{N}\sum_{i=1}^N x_i
\]  
- **Reveals**: overall average utilization.  
- **Good for**: steady patterns.  
- **Weakness**: hides bursts.

### 2. Median
\[
\text{median}(x) = \text{50th percentile of } x
\]  
- **Reveals**: central tendency robust to outliers.  
- **Good for**: noisy data with occasional spikes.

### 3. Percentiles (p95, p99)
\[
\text{p}q(x) = \min \left\{ v \,\middle|\, \frac{\#\{x_i \leq v\}}{N} \geq q \right\}
\]  
- **Reveals**: tail behavior (bursts, spikes).  
- **Good for**: detecting rare but impactful high utilization.  
- **Weakness**: insensitive to sustained high load if only a few points.

### 4. Exponential Moving Average (EMA)
\[
\text{EMA}_t = \alpha \, x_t + (1-\alpha) \, \text{EMA}_{t-1}
\]  
with smoothing factor \( \alpha \in (0,1) \).  
- **Reveals**: recent trend, weights latest samples more.  
- **Good for**: catching shifts quickly.  
- **Weakness**: sensitive to choice of α.

### 5. Coefficient of Variation (CV)
\[
\text{CV}(x) = \frac{\sigma(x)}{\mu(x)}
\]  
- **Reveals**: relative burstiness (variance scaled by mean).  
- **Good for**: identifying unstable or spiky workloads.  
- **Weakness**: undefined/unstable when mean is near zero.

### 6. Median Absolute Deviation (MAD)
\[
\text{MAD}(x) = \text{median}\bigl(|x_i - \text{median}(x)|\bigr)
\]  
- **Reveals**: robust spread of data around the median.  
- **Good for**: detecting variability without being skewed by a few outliers.  

---

## Why Combine Them?

- **Mean/Median** → summarize “central” load.  
- **Percentiles (p95/p99)** → capture rare but impactful spikes.  
- **EMA** → track recent changes in utilization.  
- **CV & MAD** → quantify burstiness and variability.  

Together, these metrics provide a **composite view** of GPU behavior.  
This allows both:
- **Pattern categorization** (Idle, Steady, Bursty, etc.), and  
- **Quantitative scoring** (e.g., a collocation risk metric).  

---
