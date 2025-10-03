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
    md = f"""# Pattern 4 — Trending (Up/Down)

**Config:** `N={params['N']}`, `ALPHA={params['ALPHA']:.6f}` (auto-derived=`{params['ALPHA_AUTO']}`)

Trends:
- SMACT: base={params['SMACT_BASE']}, slope={params['SMACT_SLOPE']}, piecewise={params['SMACT_PW']}, change_at={params['SMACT_CHG']}, slope2={params['SMACT_SLOPE2']}
- SMOCC: base={params['SMOCC_BASE']}, slope={params['SMOCC_SLOPE']}, piecewise={params['SMOCC_PW']}, change_at={params['SMOCC_CHG']}, slope2={params['SMOCC_SLOPE2']}
- DRAMA: base={params['DRAMA_BASE']}, slope={params['DRAMA_SLOPE']}, piecewise={params['DRAMA_PW']}, change_at={params['DRAMA_CHG']}, slope2={params['DRAMA_SLOPE2']}

Noise std: SMACT={params['SMACT_STD']} • SMOCC={params['SMOCC_STD']} • DRAMA={params['DRAMA_STD']}
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

# ----------------- generators -----------------
def gen_trend(N, base, slope, std, clip_min, clip_max, piecewise=False, change_at=60, slope2=0.0):
    t = np.arange(N, dtype=float)
    y = base + slope * t
    if piecewise:
        t2 = np.clip(t - change_at, 0, None)
        y += (slope2 - slope) * np.where(t >= change_at, t2, 0.0)
    y += np.random.normal(loc=0.0, scale=std, size=N)
    np.clip(y, clip_min, clip_max, out=y)
    return y

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_trending.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # config
    N = int(cfg.get("N", 120))
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    tr = cfg.get("trends", {}) or {}
    def get_trend(d, k, default): return d.get(k, default)

    S = tr.get("smact", {}); O = tr.get("smocc", {}); D = tr.get("drama", {})
    S_base, S_slope = float(get_trend(S,"base",0.2)), float(get_trend(S,"slope",0.003))
    O_base, O_slope = float(get_trend(O,"base",0.25)), float(get_trend(O,"slope",-0.002))
    D_base, D_slope = float(get_trend(D,"base",0.18)), float(get_trend(D,"slope",0.0025))
    S_pw, S_chg, S_slope2 = bool(get_trend(S,"piecewise",False)), int(get_trend(S,"change_at",60)), float(get_trend(S,"slope2",-0.002))
    O_pw, O_chg, O_slope2 = bool(get_trend(O,"piecewise",False)), int(get_trend(O,"change_at",60)), float(get_trend(O,"slope2",0.003))
    D_pw, D_chg, D_slope2 = bool(get_trend(D,"piecewise",False)), int(get_trend(D,"change_at",60)), float(get_trend(D,"slope2",-0.002))

    nz = cfg.get("noise", {}) or {}
    S_std = float(nz.get("smact_std", 0.02))
    O_std = float(nz.get("smocc_std", 0.02))
    D_std = float(nz.get("drama_std", 0.02))

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

    # synth series: linear (optionally piecewise) trend + noise
    t = np.arange(N)
    smact = gen_trend(N, S_base, S_slope, S_std, clip_min, clip_max, piecewise=S_pw, change_at=S_chg, slope2=S_slope2)
    smocc = gen_trend(N, O_base, O_slope, O_std, clip_min, clip_max, piecewise=O_pw, change_at=O_chg, slope2=O_slope2)
    drama = gen_trend(N, D_base, D_slope, D_std, clip_min, clip_max, piecewise=D_pw, change_at=D_chg, slope2=D_slope2)

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
    img_smact = os.path.join(outdir, "pattern4_smact.png")
    img_smocc = os.path.join(outdir, "pattern4_smocc.png")
    img_drama = os.path.join(outdir, "pattern4_drama.png")
    plot_series(t, smact, "Pattern 4: Trending — SMACT", img_smact)
    plot_series(t, smocc, "Pattern 4: Trending — SMOCC", img_smocc)
    plot_series(t, drama, "Pattern 4: Trending — DRAMA", img_drama)

    # README append
    section = build_md(
        {"N": N, "ALPHA": alpha, "ALPHA_AUTO": alpha_auto,
         "SMACT_BASE": S_base, "SMACT_SLOPE": S_slope, "SMACT_PW": S_pw, "SMACT_CHG": S_chg, "SMACT_SLOPE2": S_slope2,
         "SMOCC_BASE": O_base, "SMOCC_SLOPE": O_slope, "SMOCC_PW": O_pw, "SMOCC_CHG": O_chg, "SMOCC_SLOPE2": O_slope2,
         "DRAMA_BASE": D_base, "DRAMA_SLOPE": D_slope, "DRAMA_PW": D_pw, "DRAMA_CHG": D_chg, "DRAMA_SLOPE2": D_slope2,
         "SMACT_STD": S_std, "SMOCC_STD": O_std, "DRAMA_STD": D_std,
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
          {"SMACT": round(r_smact["RISK"],6), "SMOCC": round(r_smocc["RISK"],6), "DRAMA": round(r_drama["RISK"],6)})
    print(f"README {readme_mode} to: {readme_path}")

if __name__ == "__main__":
    main()
