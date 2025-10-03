#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml

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
    plt.figure(); plt.plot(t, y); plt.title(title)
    plt.xlabel("time (ticks)"); plt.ylabel("utilization")
    plt.tight_layout(); plt.savefig(outfile, dpi=140); plt.close()

def _row(d, cols): return " | ".join(f"{d[c]:.4f}" for c in cols)

def build_md(params, sS, sO, sD, rS, rO, rD, imgs):
    cols = ["mean","median","p95","p99","ema_last","cv","mad","slope"]
    md = f"""# Pattern 6 — Spike then Cool-off

**Config:** `N={params['N']}`, `ALPHA={params['ALPHA']:.6f}` (auto-derived=`{params['ALPHA_AUTO']}`)

Spike model (per metric):
- SMACT: center={params['S_C']}, width={params['S_W']}, amp={params['S_A']}, tail_start={params['S_TS']}, tau={params['S_TAU']}, tail_amp={params['S_TA']}
- SMOCC: center={params['O_C']}, width={params['O_W']}, amp={params['O_A']}, tail_start={params['O_TS']}, tau={params['O_TAU']}, tail_amp={params['O_TA']}
- DRAMA: center={params['D_C']}, width={params['D_W']}, amp={params['D_A']}, tail_start={params['D_TS']}, tau={params['D_TAU']}, tail_amp={params['D_TA']}

Baselines: SMACT={params['S_BASE']} • SMOCC={params['O_BASE']} • DRAMA={params['D_BASE']}
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
def spike_then_cool(N, base, std, center, width, amp, tail_start, tau, tail_amp, clip_min, clip_max, rng):
    t = np.arange(N, dtype=float)
    y = np.full(N, float(base), dtype=float)
    # Gaussian spike
    y += amp * np.exp(-0.5 * ((t - center) / max(1.0, float(width)))**2)
    # Exponential tail starting at tail_start
    tail = np.zeros(N, dtype=float)
    mask = t >= tail_start
    tail[mask] = tail_amp * np.exp(-(t[mask] - tail_start) / max(1.0, float(tau)))
    y += tail
    # Noise + clip
    y += rng.normal(0.0, std, size=N)
    np.clip(y, clip_min, clip_max, out=y)
    return y

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_spike_cool.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # config
    N = int(cfg.get("N", 120))
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed); rng = np.random.default_rng(seed)

    lv = cfg.get("levels", {}) or {}
    S_base = float(lv.get("smact_base", 0.10))
    O_base = float(lv.get("smocc_base", 0.08))
    D_base = float(lv.get("drama_base", 0.12))

    nz = cfg.get("noise", {}) or {}
    S_std = float(nz.get("smact_std", 0.02))
    O_std = float(nz.get("smocc_std", 0.02))
    D_std = float(nz.get("drama_std", 0.02))

    sp = cfg.get("spike", {}) or {}
    def gp(d,k,default): return d.get(k, default)
    S = sp.get("smact", {}); O = sp.get("smocc", {}); D = sp.get("drama", {})
    S_c, S_w, S_a = int(gp(S,"center",10)), float(gp(S,"width",4)), float(gp(S,"amp",0.7))
    S_ts, S_tau, S_ta = int(gp(S,"tail_start",12)), float(gp(S,"tau",15)), float(gp(S,"tail_amp",0.3))
    O_c, O_w, O_a = int(gp(O,"center",12)), float(gp(O,"width",5)), float(gp(O,"amp",0.6))
    O_ts, O_tau, O_ta = int(gp(O,"tail_start",14)), float(gp(O,"tau",18)), float(gp(O,"tail_amp",0.25))
    D_c, D_w, D_a = int(gp(D,"center",9)), float(gp(D,"width",4)), float(gp(D,"amp",0.65))
    D_ts, D_tau, D_ta = int(gp(D,"tail_start",11)), float(gp(D,"tau",16)), float(gp(D,"tail_amp",0.28))

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

    # synth series: spike early, then cool-off
    t = np.arange(N)
    smact = spike_then_cool(N, S_base, S_std, S_c, S_w, S_a, S_ts, S_tau, S_ta, clip_min, clip_max, rng)
    smocc = spike_then_cool(N, O_base, O_std, O_c, O_w, O_a, O_ts, O_tau, O_ta, clip_min, clip_max, rng)
    drama = spike_then_cool(N, D_base, D_std, D_c, D_w, D_a, D_ts, D_tau, D_ta, clip_min, clip_max, rng)

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
    img_smact = os.path.join(outdir, "pattern6_smact.png")
    img_smocc = os.path.join(outdir, "pattern6_smocc.png")
    img_drama = os.path.join(outdir, "pattern6_drama.png")
    plot_series(t, smact, "Pattern 6: Spike → Cool-off — SMACT", img_smact)
    plot_series(t, smocc, "Pattern 6: Spike → Cool-off — SMOCC", img_smocc)
    plot_series(t, drama, "Pattern 6: Spike → Cool-off — DRAMA", img_drama)

    # README append
    section = build_md(
        {"N": N, "ALPHA": alpha, "ALPHA_AUTO": alpha_auto,
         "S_C": S_c, "S_W": S_w, "S_A": S_a, "S_TS": S_ts, "S_TAU": S_tau, "S_TA": S_ta,
         "O_C": O_c, "O_W": O_w, "O_A": O_a, "O_TS": O_ts, "O_TAU": O_tau, "O_TA": O_ta,
         "D_C": D_c, "D_W": D_w, "D_A": D_a, "D_TS": D_ts, "D_TAU": D_tau, "D_TA": D_ta,
         "S_BASE": S_base, "O_BASE": O_base, "D_BASE": D_base,
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
