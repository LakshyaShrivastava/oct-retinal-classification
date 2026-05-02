"""One-off PNG generation for README figures (GitHub CDN often breaks SVG previews)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white"})

    labels = ["CNV", "DME", "DRUSEN", "NORMAL"]
    matrix = np.array(
        [
            [0.94, 0.03, 0.02, 0.01],
            [0.04, 0.92, 0.03, 0.01],
            [0.02, 0.04, 0.91, 0.03],
            [0.01, 0.02, 0.03, 0.94],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            color = "#f8f9fa" if val > 0.55 else "#1f2937"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, fontweight="600", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized (illustrative)")
    ax.set_title("Confusion matrix (illustrative — run notebook for your run)")
    plt.tight_layout()
    fig.savefig(HERE / "confusion_matrix.png", dpi=144, bbox_inches="tight")
    plt.close()

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.4), dpi=120)
    titles = ["OCT crop", "+ CAM overlay", "+ CAM overlay", "+ CAM overlay"]
    for ax, t in zip(axes, titles):
        base = rng.standard_normal((64, 64))
        cam = rng.random((64, 64))
        hm = plt.cm.hot(cam * 0.85 + base * 0.08)[..., :3]
        ax.imshow(hm)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        "Grad-CAM montage (schematic placeholders — notebook produces real overlays)",
        fontsize=10,
        y=1.06,
    )
    plt.tight_layout()
    fig.savefig(HERE / "gradcam_montage.png", dpi=144, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=120)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#252525")
    x = np.linspace(0, 8, 50)
    ax.plot(
        x,
        0.75 + 0.12 * np.sin(x * 1.4) + 0.015 * np.arange(len(x)),
        color="#569cd6",
        lw=2,
        label=r"val F$_1$ (toy)",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02,
        0.92,
        "oct-densenet · Bayes sweep (schematic)",
        transform=ax.transAxes,
        color="#cfcfcf",
        fontsize=11,
        va="top",
        fontfamily="monospace",
    )
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.legend(facecolor="#2d2d2d", edgecolor="#555555", labelcolor="#dddddd", loc="lower right")
    ax.set_xlim(0, 8)
    ax.set_ylim(0.72, 0.93)
    plt.tight_layout(pad=0.4)
    fig.savefig(HERE / "wandb_projects.png", dpi=144, bbox_inches="tight")
    plt.close()

    print("Wrote:", list(sorted(HERE.glob("*.png"))))


if __name__ == "__main__":
    main()
