#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml, random

# ----------------- helpers -----------------
def ema_last(x, alpha): return float(pd.Series(x).ewm(alpha=alpha, adjust=False).mean().iloc[-1])

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
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("time (ticks)"); plt.ylabel("utilization")
    plt.tight_layout(); plt.savefig(outfile, dpi=140); plt.close()

def _row(d, cols): return " | ".join(f"{d[c]:.4f}" for c in cols)

def build_md(params, sS, sO, sD, rS, rO, rD, imgs):
    cols = ["mean","median","p95","p99","ema_last","cv","mad","slope"]
    md = f"""# Pattern 5 — Bimodal / On–Off

**Config:** `N={params['N']}`, `ALPHA={params['ALPHA']:.6f}` (auto-derived=`{params['ALPHA_AUTO']}`)

Bimodal:
- SMACT: low={params['S_L']}, high={params['S_H']}, period={params['S_P']}, duty={params['S_D']}, jitter={params['S_J']}, random_phase={params['S_RP']}
- SMOCC: low={params['O_L']}, high={params['O_H']}, period={params['O_P']}, duty={params['O_D']}, jitter={params['O_J']}, random_phase={params['O_RP']}
- DRAMA: low={params['D_L']}, high={params['D_H']}, period={params['D_P']}, duty={params['D_D']}, jitter={params['D_J']}, random_phase={params['D_RP']}

Noise std: SMACT={params['S_STD']} • SMOCC={params['O_STD']} • DRAMA={params['D_STD']}
Clip: [{params['CLIP_MIN']}, {params['CLIP_MAX']}]

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

# ----------------- generator -----------------
def gen_bimodal(N, low, high, period, duty, jitter, random_phase, std, clip_min, clip_max, rng):
    t = np.arange(N)
    if period <= 0: period = 1
    on_len = max(0, min(period, int(round(period * duty))))
    off_len = max(0, period - on_len)

    # choose start phase
    start_offset = rng.integers(0, period) if random_phase else 0

    y = np.empty(N, dtype=float)
    idx = 0
    # build cycles
    while idx < N:
        # optional jitter on boundaries
        j_on = int(rng.integers(-jitter, jitter+1)) if jitter > 0 else 0
        j_off = int(rng.integers(-jitter, jitter+1)) if jitter > 0 else 0
        # HIGH phase
        k_on = max(0, min(on_len + j_on, N - idx))
        if k_on > 0:
            y[idx:idx+k_on] = high
            idx += k_on
        # LOW phase
        k_off = max(0, min(off_len + j_off, N - idx))
        if k_off > 0:
            y[idx:idx+k_off] = low
            idx += k_off

    # apply start offset rotation
    if start_offset > 0:
        y = np.roll(y, start_offset)

    # add noise and clip
    y = y + rng.normal(0.0, std, size=N)
    np.clip(y, clip_min, clip_max, out=y)
    return y

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_bimodal.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # config
    N = int(cfg.get("N", 120))
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    bm = cfg.get("bimodal", {}) or {}
    def getb(d, k, default): return d.get(k, default)

    S = bm.get("smact", {}); O = bm.get("smocc", {}); D = bm.get("drama", {})
    S_low, S_high = float(getb(S,"low",0.1)), float(getb(S,"high",0.8))
    O_low, O_high = float(getb(O,"low",0.08)), float(getb(O,"high",0.6))
    D_low, D_high = float(getb(D,"low",0.12)), float(getb(D,"high",0.7))
    S_p, S_d, S_j, S_rp = int(getb(S,"period",20)), float(getb(S,"duty_cycle",0.5)), int(getb(S,"jitter",0)), bool(getb(S,"random_phase",True))
    O_p, O_d, O_j, O_rp = int(getb(O,"period",24)), float(getb(O,"duty_cycle",0.5)), int(getb(O,"jitter",0)), bool(getb(O,"random_phase",True))
    D_p, D_d, D_j, D_rp = int(getb(D,"period",22)), float(getb(D,"duty_cycle",0.5)), int(getb(D,"jitter",0)), bool(getb(D,"random_phase",True))

    nz = cfg.get("noise", {}) or {}
    S_std = float(nz.get("smact_std", 0.03))
    O_std = float(nz.get("smocc_std", 0.03))
    D_std = float(nz.get("drama_std", 0.03))

    clip_min = float(cfg.get("clip_min", 0.0))
    clip_max = float(cfg.get("clip_max", 1.0))

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

    # synth series: on-off per metric
    t = np.arange(N)
    smact = gen_bimodal(N, S_low, S_high, S_p, S_d, S_j, S_rp, S_std, clip_min, clip_max, rng)
    smocc = gen_bimodal(N, O_low, O_high, O_p, O_d, O_j, O_rp, O_std, clip_min, clip_max, rng)
    drama = gen_bimodal(N, D_low, D_high, D_p, D_d, D_j, D_rp, D_std, clip_min, clip_max, rng)

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
    img_smact = os.path.join(outdir, "pattern5_smact.png")
    img_smocc = os.path.join(outdir, "pattern5_smocc.png")
    img_drama = os.path.join(outdir, "pattern5_drama.png")
    plot_series(t, smact, "Pattern 5: Bimodal/On–Off — SMACT", img_smact)
    plot_series(t, smocc, "Pattern 5: Bimodal/On–Off — SMOCC", img_smocc)
    plot_series(t, drama, "Pattern 5: Bimodal/On–Off — DRAMA", img_drama)

    # README append
    section = build_md(
        {"N": N, "ALPHA": alpha, "ALPHA_AUTO": alpha_auto,
         "S_L": S_low, "S_H": S_high, "S_P": S_p, "S_D": S_d, "S_J": S_j, "S_RP": S_rp,
         "O_L": O_low, "O_H": O_high, "O_P": O_p, "O_D": O_d, "O_J": O_j, "O_RP": O_rp,
         "D_L": D_low, "D_H": D_high, "D_P": D_p, "D_D": D_d, "D_J": D_j, "D_RP": D_rp,
         "S_STD": S_std, "O_STD": O_std, "D_STD": D_std,
         "CLIP_MIN": clip_min, "CLIP_MAX": clip_max,
         "wT": wT, "wE": wE, "wB": wB, "wC": wC},
        s_smact, s_smocc, s_drama,
        r_smact, r_smocc, r_drama,
        {"smact": img_smact, "smocc": img_smocc, "drama": img_drama}
    )
    readme_path = os.path.join(outdir, readme)
    write_readme(readme_path, section, mode=readme_mode)

    print(f"alpha = {alpha:.6f} ({'auto' if alpha_auto else 'manual'})")
    print("Per-metric RISK:",
          {"SMACT": round(r_smact['RISK'],6), "SMOCC": round(r_smocc['RISK'],6), "DRAMA": round(r_drama['RISK'],6)})
    print(f"README {readme_mode} to: {readme_path}")

if __name__ == "__main__":
    main()
