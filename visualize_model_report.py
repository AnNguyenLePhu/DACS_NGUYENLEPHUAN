import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_DPI = 160


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _load_metrics(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "merged_metrics_ALL.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _bar_train_val_test_mape(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["Tr_MAPE", "Va_MAPE", "Te_MAPE", "Naive_MAPE"]
    if any(c not in df.columns for c in cols):
        return

    grouped = df.groupby(["Scenario", "Model"])[cols].mean().reset_index()
    labels = grouped["Scenario"] + " | " + grouped["Model"]
    x = np.arange(len(grouped))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, col in enumerate(cols):
        ax.bar(x + (idx - 1.5) * width, grouped[col], width, label=col)
    ax.set_title("Train / Validation / Test MAPE vs Naive")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir / "01_mape_train_val_test_vs_naive.png")


def _bar_generalization_gaps(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in ["TrainTest_R2_gap", "ValTest_R2_gap", "MAPE_ValTest_gap"] if c in df.columns]
    if not cols:
        return

    grouped = df.groupby(["Scenario", "Model"])[cols].mean().reset_index()
    labels = grouped["Scenario"] + " | " + grouped["Model"]
    x = np.arange(len(grouped))
    width = 0.8 / len(cols)

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, col in enumerate(cols):
        ax.bar(x + (idx - (len(cols) - 1) / 2) * width, grouped[col], width, label=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Generalization Gaps for Overfitting Diagnosis")
    ax.set_ylabel("Gap value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(ncol=len(cols))
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir / "02_overfit_generalization_gaps.png")


def _scatter_copy_lazy(df: pd.DataFrame, out_dir: Path) -> None:
    needed = {"MoveRatio", "CopyRatio", "DA_test", "NaiveMAPERatio", "Scenario", "Model"}
    if not needed.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["MoveRatio"],
        df["CopyRatio"],
        c=df["DA_test"],
        s=np.clip(df["NaiveMAPERatio"], 0.5, 2.0) * 80,
        alpha=0.75,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.7,
    )
    ax.axvline(0.35, color="#d62728", linestyle="--", linewidth=1, label="MoveRatio 0.35")
    ax.axhline(0.60, color="#9467bd", linestyle="--", linewidth=1, label="CopyRatio 0.60")
    ax.set_title("Lazy / Copy Diagnosis: MoveRatio vs CopyRatio")
    ax.set_xlabel("MoveRatio: mean(|Pred - LastClose|) / mean(|Actual - LastClose|)")
    ax.set_ylabel("CopyRatio: % predictions near LastClose")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Directional Accuracy (%)")
    _save(fig, out_dir / "03_lazy_copy_scatter.png")


def _heatmap_metric(df: pd.DataFrame, out_dir: Path, metric: str, filename: str, title: str) -> None:
    if metric not in df.columns:
        return
    pivot = df.pivot_table(index="Scenario", columns="Model", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    _save(fig, out_dir / filename)


def _prediction_plots(results_dir: Path, out_dir: Path, max_plots: int) -> None:
    pred_files = sorted(results_dir.glob("*/*/pred_*_scenario_*.csv"))
    if max_plots > 0:
        pred_files = pred_files[:max_plots]

    pred_out = out_dir / "predictions"
    _ensure_dir(pred_out)
    for pred_path in pred_files:
        df = pd.read_csv(pred_path)
        if not {"Date", "Actual", "Predicted"}.issubset(df.columns):
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        model = pred_path.name.replace("pred_", "").replace(".csv", "")
        scenario = pred_path.parent.name
        ticker = pred_path.parent.parent.name

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2.2, 1])
        axes[0].plot(df["Date"], df["Actual"], label="Actual", linewidth=1.8)
        axes[0].plot(df["Date"], df["Predicted"], label="Predicted", linewidth=1.5)
        axes[0].set_title(f"{ticker} | {scenario} | {model}: Actual vs Predicted")
        axes[0].set_ylabel("Close")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        error = df["Predicted"] - df["Actual"]
        axes[1].bar(df["Date"], error, width=1.0, color=np.where(error >= 0, "#2ca02c", "#d62728"))
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Pred - Actual")
        axes[1].grid(axis="y", alpha=0.25)
        _save(fig, pred_out / f"{ticker}_{scenario}_{model}.png")


def _loss_curves(results_dir: Path, out_dir: Path, max_plots: int) -> None:
    history_files = sorted(results_dir.glob("*/*/history_*_scenario_*.json"))
    if max_plots > 0:
        history_files = history_files[:max_plots]

    hist_out = out_dir / "loss_curves"
    _ensure_dir(hist_out)
    for history_path in history_files:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if "loss" not in history or "val_loss" not in history:
            continue
        ticker = history_path.parent.parent.name
        scenario = history_path.parent.name
        model = history_path.name.replace("history_", "").replace(".json", "")
        epochs = np.arange(1, len(history["loss"]) + 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, history["loss"], label="train_loss", linewidth=1.8)
        ax.plot(epochs, history["val_loss"], label="val_loss", linewidth=1.8)
        ax.set_title(f"{ticker} | {scenario} | {model}: Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, hist_out / f"{ticker}_{scenario}_{model}.png")


def create_report(results_dir: str, max_detail_plots: int) -> Path:
    results_path = Path(results_dir)
    out_dir = results_path / "report_charts"
    _ensure_dir(out_dir)

    df = _load_metrics(results_path)
    _bar_train_val_test_mape(df, out_dir)
    _bar_generalization_gaps(df, out_dir)
    _scatter_copy_lazy(df, out_dir)
    _heatmap_metric(df, out_dir, "MoveRatio", "04_heatmap_moveratio.png", "Average MoveRatio")
    _heatmap_metric(df, out_dir, "CopyRatio", "05_heatmap_copyratio.png", "Average CopyRatio")
    _heatmap_metric(df, out_dir, "DA_test", "06_heatmap_directional_accuracy.png", "Average Directional Accuracy")
    _heatmap_metric(df, out_dir, "NaiveImprove_MAPE", "07_heatmap_naive_improve_mape.png", "Average MAPE Improvement vs Naive")
    _prediction_plots(results_path, out_dir, max_detail_plots)
    _loss_curves(results_path, out_dir, max_detail_plots)

    summary_cols = [
        "Tr_MAPE", "Va_MAPE", "Te_MAPE", "Naive_MAPE",
        "NaiveImprove_MAPE", "MoveRatio", "CopyRatio", "LazyRatio",
        "ZeroReturnRatio", "LastCloseCorr", "DA_test",
        "TrainTest_R2_gap", "ValTest_R2_gap", "MAPE_ValTest_gap",
        "ReturnCorr_test",
    ]
    existing = [c for c in summary_cols if c in df.columns]
    if existing:
        summary = df.groupby(["Scenario", "Model"])[existing].mean(numeric_only=True).round(4)
        summary.to_csv(out_dir / "visual_report_summary.csv")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual diagnostic charts for VN30 model reports.")
    parser.add_argument("--results-dir", default="D:/DACS/3.0/results")
    parser.add_argument(
        "--max-detail-plots",
        type=int,
        default=30,
        help="Maximum prediction/loss detail charts. Use 0 for all.",
    )
    args = parser.parse_args()
    out_dir = create_report(args.results_dir, args.max_detail_plots)
    print(f"Saved report charts to: {out_dir}")


if __name__ == "__main__":
    main()
