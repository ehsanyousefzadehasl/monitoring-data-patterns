#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml, random

# ----------------- helpers (same as before) -----------------
def ema_last(x, alpha):
    return float(pd.Series(x).ewm(alpha=alpha, adjust=False).mean().iloc[-1])

def cv(x):
    x = np.asarray(x, dtype=float); m = x.mean()
    return float(x.std(ddof=0) / m) if m > 0 else float("nan")

def mad(x):
    x = np.asarray(x, dtype=float); med = np.median(x)
    return float(np.median(np.abs(x - med)))

def pctl(x, q): return float(np.quantile(np.asarray(x, dtype=float), q))

def lin_slope(x):
    y = np.asarray(x, dtype=float); t = np.arange(len(y), dtype=float)
    tc = t - t.mean(); denom = float((tc**2).sum())
    if denom == 0.0: return 0.0
    return float((tc * (y - y.mean())).sum() / denom)

def trend_flag(x, thresh): return 1 if abs(lin_slope(x)) > float(thresh) else 0

def stats_series(x, alpha, slope_thresh):
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": pctl(x, 0.95),
        "p99": pctl(x, 0.99),
        "ema_last": ema_last(x, alpha),
        "cv": cv(x),
        "mad": mad(x),
        "slope": lin_slope(x),
        "trend_flag": trend_flag(x, slope_thresh),
    }

def risk_per_metric(stats, wT, wE, wB, wC):
    T = stats["p95"]; E = stats["ema_last"]; B = stats["cv"]; C = float(bool(stats["trend_flag"]))
    return {"T": T, "E": E, "B": B, "C": C, "RISK": wT*T + wE*E + wB*B + wC*C}

def plot_series(t, y, title, outfile):
    plt.figure(); plt.plot(t, y); plt.title(title)
    plt.xlabel("time (ticks)"); plt.ylabel("utilization")
    plt.tight_layout(); plt.savefig(outfile, dpi=140); plt.close()

def _row(d, cols): return " | ".join(f"{d[c]:.4f}" for c in cols)

def build_md(params, sS, sO, sD, rS, rO, rD, imgs):
    cols = ["mean","median","p95","p99","ema_last","cv","mad","slope"]
    md = f"""# Pattern 3 — Bursty / Spiky

**Config:** `N={params['N']}`, `ALPHA={params['ALPHA']:.6f}` (auto-derived=`{params['ALPHA_AUTO']}`)  
Baselines: SMACT={params['SMACT_BASE']} • SMOCC={params['SMOCC_BASE']} • DRAMA={params['DRAMA_BASE']}  
Noise std: SMACT={params['SMACT_STD']} • SMOCC={params['SMOCC_STD']} • DRAMA={params['DRAMA_STD']}  
Bursts (λ): SMACT={params['B_SM_LAM']} • SMOCC={params['B_SO_LAM']} • DRAMA={params['B_DR_LAM']}

## Plots
![SMACT]({os.path.basename(imgs['smact'])})
![SMOCC]({os.path.basename(imgs['smocc'])})
![DRAMA]({os.path.basename(imgs['drama'])})

## Window Statistics (per metric)
Metric | mean | median | p95 | p99 | EMA_last | CV | MAD | slope
---|---:|---:|---:|---:|---:|---:|---:|---:
SMACT | {_row(sS, cols)}
SMOCC | {_row(sO, cols)}
DRAMA | {_row(sD, cols)}

Trend flags: SMACT={sS['trend_flag']} • SMOCC={sO['trend_flag']} • DRAMA={sD['trend_flag']}

## Per-Metric Risk (no mixing)
Weights: wT={params['wT']}, wE={params['wE']}, wB={params['wB']}, wC={params['wC']}

Metric | T (p95) | E (EMA) | B (CV) | C (trend) | RISK
---|---:|---:|---:|---:|---:
SMACT | {rS['T']:.4f} | {rS['E']:.4f} | {rS['B']:.4f} | {rS['C']:.1f} | {rS['RISK']:.4f}
SMOCC | {rO['T']:.4f} | {rO['E']:.4f} | {rO['B']:.4f} | {rO['C']:.1f} | {rO['RISK']:.4f}
DRAMA | {rD['T']:.4f} | {rD['E']:.4f} | {rD['B']:.4f} | {rD['C']:.1f} | {rD['RISK']:.4f}
"""
    return md

def write_readme(path, content, mode="append"):
    if mode == "append" and os.path.exists(path):
        with open(path, "a", encoding="utf-8") as f: f.write("\n\n" + content)
    else:
        with open(path, "w", encoding="utf-8") as f: f.write(content)

# ----------------- burst generator -----------------
def add_gaussian_bursts(y, lam, amp_min, amp_max, width_min, width_max, rng):
    N = len(y)
    k = rng.poisson(lam=lam)  # how many bursts
    for _ in range(k):
        center = rng.integers(low=0, high=N)
        amp = rng.uniform(amp_min, amp_max)
        width = rng.integers(width_min, width_max+1)
        # Gaussian pulse
        t = np.arange(N)
        pulse = amp * np.exp(-0.5 * ((t - center) / max(1, width))**2)
        y += pulse
    # clip to [0,1]
    np.clip(y, 0.0, 1.0, out=y)
    return y

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_bursty.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # config
    N = int(cfg.get("N", 120))
    seed = int(cfg.get("seed", 42))

    lv = cfg.get("levels", {}) or {}
    smact_base = float(lv.get("smact_base", 0.15))
    smocc_base = float(lv.get("smocc_base", 0.10))
    drama_base = float(lv.get("drama_base", 0.12))

    nz = cfg.get("noise", {}) or {}
    smact_std = float(nz.get("smact_std", 0.03))
    smocc_std = float(nz.get("smocc_std", 0.03))
    drama_std = float(nz.get("drama_std", 0.03))

    bcfg = cfg.get("bursts", {}) or {}
    bS = bcfg.get("smact", {}); bO = bcfg.get("smocc", {}); bD = bcfg.get("drama", {})
    # defaults
    def bp(d, key, default): return d.get(key, default)
    lamS = float(bp(bS, "lam", 2.0)); lamO = float(bp(bO, "lam", 2.0)); lamD = float(bp(bD, "lam", 2.0))
    S = dict(amp_min=float(bp(bS,"amp_min",0.4)), amp_max=float(bp(bS,"amp_max",0.8)),
             width_min=int(bp(bS,"width_min",3)), width_max=int(bp(bS,"width_max",8)))
    O = dict(amp_min=float(bp(bO,"amp_min",0.35)), amp_max=float(bp(bO,"amp_max",0.7)),
             width_min=int(bp(bO,"width_min",3)), width_max=int(bp(bO,"width_max",8)))
    D = dict(amp_min=float(bp(bD,"amp_min",0.35)), amp_max=float(bp(bD,"amp_max",0.7)),
             width_min=int(bp(bD,"width_min",3)), width_max=int(bp(bD,"width_max",8)))

    alpha_mode = str(cfg.get("alpha_mode", "auto")).lower()
    alpha_cfg = float(cfg.get("alpha", 0.1))
    if alpha_mode == "manual":
        alpha = alpha_cfg; alpha_auto = False
    else:
        alpha = 2.0 / (N + 1.0); alpha_auto = True

    outdir = cfg.get("outdir", ".")
    readme = cfg.get("readme", "README.md")
    readme_mode = str(cfg.get("readme_mode", "append")).lower()

    w = cfg.get("risk_weights", {}) or {}
    wT, wE, wB, wC = float(w.get("wT", 0.5)), float(w.get("wE", 0.3)), float(w.get("wB", 0.1)), float(w.get("wC", 0.1))
    slope_thresh = float(cfg.get("trend_slope_threshold", 0.002))

    # RNGs
    np.random.seed(seed)
    rng_np = np.random.default_rng(seed)

    # synth series: baseline + noise + bursts
    t = np.arange(N)
    smact = np.clip(np.random.normal(loc=smact_base, scale=smact_std, size=N), 0, 1)
    smocc = np.clip(np.random.normal(loc=smocc_base, scale=smocc_std, size=N), 0, 1)
    drama = np.clip(np.random.normal(loc=drama_base, scale=drama_std, size=N), 0, 1)

    smact = add_gaussian_bursts(smact, lamS, **S, rng=rng_np)
    smocc = add_gaussian_bursts(smocc, lamO, **O, rng=rng_np)
    drama = add_gaussian_bursts(drama, lamD, **D, rng=rng_np)

    # stats per metric
    s_smact = stats_series(smact, alpha, slope_thresh)
    s_smocc = stats_series(smocc, alpha, slope_thresh)
    s_drama = stats_series(drama, alpha, slope_thresh)

    # per-metric risk
    r_smact = risk_per_metric(s_smact, wT, wE, wB, wC)
    r_smocc = risk_per_metric(s_smocc, wT, wE, wB, wC)
    r_drama = risk_per_metric(s_drama, wT, wE, wB, wC)

    # plots
    os.makedirs(outdir, exist_ok=True)
    img_smact = os.path.join(outdir, "pattern3_smact.png")
    img_smocc = os.path.join(outdir, "pattern3_smocc.png")
    img_drama = os.path.join(outdir, "pattern3_drama.png")
    plot_series(t, smact, "Pattern 3: Bursty/Spiky — SMACT", img_smact)
    plot_series(t, smocc, "Pattern 3: Bursty/Spiky — SMOCC", img_smocc)
    plot_series(t, drama, "Pattern 3: Bursty/Spiky — DRAMA", img_drama)

    # README append
    section = build_md(
        {"N": N, "ALPHA": alpha, "ALPHA_AUTO": alpha_auto,
         "SMACT_BASE": smact_base, "SMOCC_BASE": smocc_base, "DRAMA_BASE": drama_base,
         "SMACT_STD": smact_std, "SMOCC_STD": smocc_std, "DRAMA_STD": drama_std,
         "B_SM_LAM": lamS, "B_SO_LAM": lamO, "B_DR_LAM": lamD,
         "wT": wT, "wE": wE, "wB": wB, "wC": wC},
        s_smact, s_smocc, s_drama,
        r_smact, r_smocc, r_drama,
        {"smact": img_smact, "smocc": img_smocc, "drama": img_drama}
    )
    readme_path = os.path.join(outdir, readme)
    write_readme(readme_path, section, mode=readme_mode)

    print(f"alpha = {alpha:.6f} ({'auto' if alpha_auto else 'manual'})")
    print("Per-metric RISK:",
          {"SMACT": round(r_smact["RISK"],6), "SMOCC": round(r_smocc["RISK"],6), "DRAMA": round(r_drama["RISK"],6)})
    print(f"README {readme_mode} to: {readme_path}")

if __name__ == "__main__":
    main()
