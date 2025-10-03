# pattern1_idle_flatlow.py
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

def ema_last(x, alpha=0.1):
    s = pd.Series(x).ewm(alpha=alpha, adjust=False).mean()
    return float(s.iloc[-1])

def cv(x):
    x = np.asarray(x, dtype=float); m = x.mean()
    return float(x.std(ddof=0) / m) if m > 0 else float("nan")

def mad(x):
    x = np.asarray(x, dtype=float); med = np.median(x)
    return float(np.median(np.abs(x - med)))

def stats(x, alpha):
    x = np.asarray(x, dtype=float)
    return {
        "mean": x.mean(),
        "median": np.median(x),
        "p95": np.quantile(x, 0.95),
        "p99": np.quantile(x, 0.99),
        "ema_last": ema_last(x, alpha=alpha),
        "cv": cv(x),
        "mad": mad(x),
    }

def plot_series(t, y, title, outfile):
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("time (ticks)")
    plt.ylabel("utilization")
    plt.tight_layout()
    plt.savefig(outfile, dpi=140)
    plt.close()

def write_readme(path, params, smact_stats, smocc_stats, drama_stats, img_paths):
    def row(d):
        return f"{d['mean']:.4f} | {d['median']:.4f} | {d['p95']:.4f} | {d['p99']:.4f} | {d['ema_last']:.4f} | {d['cv']:.4f} | {d['mad']:.4f}"
    md = f"""# Pattern 1 — Idle / Flat Low

**Config:** `N={params['N']}`, `NOISE={params['NOISE']}`, `ALPHA={params['ALPHA']}`

## Plots
![SMACT]({os.path.basename(img_paths['smact'])})
![SMOCC]({os.path.basename(img_paths['smocc'])})
![DRAMA]({os.path.basename(img_paths['drama'])})

## Window Statistics
Metric | mean | median | p95 | p99 | EMA_last | CV | MAD
---|---:|---:|---:|---:|---:|---:|---:
SMACT | {row(smact_stats)}
SMOCC | {row(smocc_stats)}
DRAMA | {row(drama_stats)}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=120, help="samples in window")
    ap.add_argument("--noise", type=float, default=0.01, help="noise std around near-zero")
    ap.add_argument("--alpha", type=float, default=0.1, help="EMA smoothing factor")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default=".", help="output directory")
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    t = np.arange(args.N)
    smact = np.clip(np.random.normal(loc=0.02, scale=args.noise, size=args.N), 0, 1)
    smocc = np.clip(np.random.normal(loc=0.01, scale=args.noise, size=args.N), 0, 1)
    drama = np.clip(np.random.normal(loc=0.015, scale=args.noise, size=args.N), 0, 1)

    smact_s = stats(smact, args.alpha)
    smocc_s = stats(smocc, args.alpha)
    drama_s = stats(drama, args.alpha)

    img_smact = os.path.join(args.outdir, "pattern1_smact.png")
    img_smocc = os.path.join(args.outdir, "pattern1_smocc.png")
    img_drama = os.path.join(args.outdir, "pattern1_drama.png")

    plot_series(t, smact, "Pattern 1: Idle/Flat Low — SMACT", img_smact)
    plot_series(t, smocc, "Pattern 1: Idle/Flat Low — SMOCC", img_smocc)
    plot_series(t, drama, "Pattern 1: Idle/Flat Low — DRAMA", img_drama)

    readme = os.path.join(args.outdir, "README.md")
    write_readme(
        readme,
        {"N": args.N, "NOISE": args.noise, "ALPHA": args.alpha},
        smact_s, smocc_s, drama_s,
        {"smact": img_smact, "smocc": img_smocc, "drama": img_drama},
    )

    # Also print stats to console (concise)
    for name, s in [("SMACT", smact_s), ("SMOCC", smocc_s), ("DRAMA", drama_s)]:
        print(name, {k: round(v, 6) for k, v in s.items()})

if __name__ == "__main__":
    main()
