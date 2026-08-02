import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 11})

bccd_bdqi = [99.67, 93.82, 85.86, 70.76, 69.12]
bccd_map = [0.9458, 0.9453, 0.9389, 0.9239, 0.9037]
chula_bdqi = [99.78, 81.76, 72.70, 70.63, 68.78]
chula_map = [0.5607, 0.5568, 0.5299, 0.4610, 0.3743]
levels = ["Clean", "Mild", "Moderate", "Severe", "Extreme"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

for ax, bdqi, mapv, title, color in [
    (ax1, bccd_bdqi, bccd_map, "BCCD", "#4C72B0"),
    (ax2, chula_bdqi, chula_map, "Chula_RBC", "#DD8452"),
]:
    ax.scatter(bdqi, mapv, s=90, color=color, edgecolor="#333333", linewidth=1.2, zorder=3)
    # connect points in degradation order to show the trajectory
    ax.plot(bdqi, mapv, color=color, alpha=0.4, linewidth=1.5, zorder=2, linestyle="--")
    for x, y, lbl in zip(bdqi, mapv, levels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8.5)

    # linear fit line for visual reference (not implying linearity is assumed elsewhere)
    z = np.polyfit(bdqi, mapv, 1)
    xs = np.linspace(min(bdqi) - 2, max(bdqi) + 2, 50)
    ax.plot(xs, np.polyval(z, xs), color="#888888", linewidth=1, linestyle=":", zorder=1)

    r = np.corrcoef(bdqi, mapv)[0, 1]
    ax.set_title(f"{title}  (Pearson r = {r:.2f}, Spearman \u03c1 = 1.00)", fontsize=11)
    ax.set_xlabel("BDQI")
    ax.set_ylabel("Test mAP50")
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle("Figure 6. BDQI vs. test mAP50 across five degradation levels, BCCD and Chula_RBC side by side",
             fontsize=12.5, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("/home/claude/review_figures/Figure6_BDQI_Scatter.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("BDQI scatter plot saved")
