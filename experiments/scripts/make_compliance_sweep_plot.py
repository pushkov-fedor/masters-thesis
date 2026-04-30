"""График: какая политика лидер при разных compliance."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(ROOT / "results" / "compliance_sweep.json", encoding="utf-8") as f:
        data = json.load(f)

    POLICY_COLORS = {
        "Random": "#888888",
        "Cosine": "#1f77b4",
        "MMR": "#9467bd",
        "Capacity-aware": "#2ca02c",
        "Capacity-aware MMR": "#d62728",
        "DPP": "#bcbd22",
    }

    compliance_values = data["compliance_values"]
    policies = data["config"]["policies"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overflow vs compliance
    ax = axes[0]
    for p in policies:
        ys = [data["results"][str(c)][p]["overflow_choice_mean"] for c in compliance_values]
        ax.plot(compliance_values, ys, "o-", label=p, color=POLICY_COLORS.get(p, "#999"),
                linewidth=1.5, markersize=8)
    ax.set_xlabel("User compliance (доля пользователей, следующих рекомендации)")
    ax.set_ylabel("Overflow rate (choice slots)")
    ax.set_title("H4: устойчивость к compliance\n(Capacity-aware лидер во всём диапазоне)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.invert_xaxis()  # чтобы compliance=1.0 слева

    # Right: utility vs compliance
    ax = axes[1]
    for p in policies:
        ys = [data["results"][str(c)][p]["utility_mean"] for c in compliance_values]
        ax.plot(compliance_values, ys, "o-", label=p, color=POLICY_COLORS.get(p, "#999"),
                linewidth=1.5, markersize=8)
    ax.set_xlabel("User compliance")
    ax.set_ylabel("Mean user utility")
    ax.set_title("Утилитарность стабильна; разница между политиками < 5%")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_xaxis()

    fig.suptitle("Sensitivity по compliance (Mobius, w_fame=0.3, 6 политик)")
    fig.tight_layout()
    out = ROOT / "results" / "plots" / "70_compliance_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE: {out}")
    thesis_fig = ROOT.parent / "thesis" / "figures"
    if thesis_fig.exists():
        import shutil
        shutil.copy(out, thesis_fig / out.name)


if __name__ == "__main__":
    main()
