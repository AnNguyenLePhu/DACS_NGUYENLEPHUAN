import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import main
from config import FEATURE_COLS, MODEL_NAMES, RESULTS_DIR, SCENARIOS


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _log(message: str) -> None:
    print(message, flush=True)


def _scenario_by_name() -> dict:
    return {scenario["name"]: scenario for scenario in SCENARIOS}


def _load_split(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["TradingDate"] = pd.to_datetime(df["TradingDate"], errors="coerce")
    return df


def _build_windows_from_saved_artifacts(folder: Path, scenario: dict) -> dict:
    train_df = _load_split(folder, "train")
    val_df = _load_split(folder, "val")
    test_df = _load_split(folder, "test")

    scaler_x_path = folder / "scaler_x.pkl"
    scaler_y_path = folder / "scaler_y.pkl"
    if not scaler_x_path.exists():
        raise FileNotFoundError(scaler_x_path)
    if not scaler_y_path.exists():
        raise FileNotFoundError(scaler_y_path)

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    tr_scaled = scaler_x.transform(train_df[FEATURE_COLS].values)
    va_scaled = scaler_x.transform(val_df[FEATURE_COLS].values)
    te_scaled = scaler_x.transform(test_df[FEATURE_COLS].values)

    raw_tr = train_df["Close"].values.astype(np.float64)
    raw_va = val_df["Close"].values.astype(np.float64)
    raw_te = test_df["Close"].values.astype(np.float64)

    tr_dates = train_df["TradingDate"].values
    va_dates = val_df["TradingDate"].values
    te_dates = test_df["TradingDate"].values

    lookback = scenario["lookback"]
    horizon = scenario["horizon"]

    X_tr, Y_tr, Yd_tr, Lc_tr = main._make_windows(
        tr_scaled, raw_tr, tr_dates, lookback, horizon
    )
    X_va, Y_va, Yd_va, Lc_va = main._make_windows(
        va_scaled,
        raw_va,
        va_dates,
        lookback,
        horizon,
        context_scaled=tr_scaled[-lookback:],
        context_close=raw_tr[-lookback:],
        context_dates=tr_dates[-lookback:],
    )
    X_te, Y_te, Yd_te, Lc_te = main._make_windows(
        te_scaled,
        raw_te,
        te_dates,
        lookback,
        horizon,
        context_scaled=va_scaled[-lookback:],
        context_close=raw_va[-lookback:],
        context_dates=va_dates[-lookback:],
    )

    if X_tr is None or X_va is None or X_te is None:
        raise ValueError(f"Not enough windows in {folder}")

    return {
        "X_tr": X_tr,
        "Y_tr": Y_tr,
        "Yd_tr": Yd_tr,
        "Lc_tr": Lc_tr,
        "X_va": X_va,
        "Y_va": Y_va,
        "Yd_va": Yd_va,
        "Lc_va": Lc_va,
        "X_te": X_te,
        "Y_te": Y_te,
        "Yd_te": Yd_te,
        "Lc_te": Lc_te,
        "scaler_y": scaler_y,
    }


def _predict(model: tf.keras.Model, x: np.ndarray, use_mc_dropout: bool) -> np.ndarray:
    if use_mc_dropout:
        return main._mc_predict(model, x)
    return model.predict(x, verbose=0)


def _evaluate_model(
    model_path: Path,
    ticker: str,
    scenario: dict,
    model_name: str,
    windows: dict,
    use_mc_dropout: bool,
    write_predictions: bool,
    folder: Path,
) -> dict:
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler_y = windows["scaler_y"]
    horizon = scenario["horizon"]
    scenario_name = scenario["name"]

    pred_tr_s = _predict(model, windows["X_tr"], use_mc_dropout)
    pred_va_s = _predict(model, windows["X_va"], use_mc_dropout)
    pred_te_s = _predict(model, windows["X_te"], use_mc_dropout)

    act_tr = main._actual_price(windows["Y_tr"], windows["Lc_tr"])
    act_va = main._actual_price(windows["Y_va"], windows["Lc_va"])
    act_te = main._actual_price(windows["Y_te"], windows["Lc_te"])

    return_scale = main._select_return_scale(
        pred_va_s, act_va, windows["Lc_va"], scaler_y, horizon
    )
    pred_tr = main._inverse_to_price(
        pred_tr_s, windows["Lc_tr"], scaler_y, return_scale=return_scale
    )
    pred_va = main._inverse_to_price(
        pred_va_s, windows["Lc_va"], scaler_y, return_scale=return_scale
    )
    pred_te = main._inverse_to_price(
        pred_te_s, windows["Lc_te"], scaler_y, return_scale=return_scale
    )

    m_tr = main._metrics(act_tr, pred_tr)
    m_va = main._metrics(act_va, pred_va)
    m_te = main._metrics(act_te, pred_te)
    r_tr = main._return_metrics(act_tr, pred_tr, windows["Lc_tr"], horizon)
    r_va = main._return_metrics(act_va, pred_va, windows["Lc_va"], horizon)
    r_te = main._return_metrics(act_te, pred_te, windows["Lc_te"], horizon)

    m_tr["DA"] = main._directional_accuracy(act_tr, pred_tr, windows["Lc_tr"])
    m_va["DA"] = main._directional_accuracy(act_va, pred_va, windows["Lc_va"])
    m_te["DA"] = main._directional_accuracy(act_te, pred_te, windows["Lc_te"])

    naive = main._naive_metrics(act_te, windows["Lc_te"], horizon)
    anti_lazy = main._anti_lazy_metrics(
        act_te, pred_te, windows["Lc_te"], naive, m_te["MAPE"], horizon
    )

    ev = {
        "ticker": ticker,
        "model": model_name,
        "scenario": scenario_name,
        "m_tr": m_tr,
        "m_va": m_va,
        "m_te": m_te,
        "r_tr": r_tr,
        "r_va": r_va,
        "r_te": r_te,
        "naive": naive,
        "anti_lazy": anti_lazy,
        "pred_tr": pred_tr,
        "act_tr": act_tr,
        "Lc_tr": windows["Lc_tr"],
        "pred_va": pred_va,
        "act_va": act_va,
        "Lc_va": windows["Lc_va"],
        "pred_te": pred_te,
        "act_te": act_te,
        "Yd_te": windows["Yd_te"],
        "Lc_te": windows["Lc_te"],
    }
    audit = main.step11_overfitting_audit(ev)

    if write_predictions:
        pred_df = main._build_pred_df(windows["Yd_te"], act_te, pred_te, horizon)
        pred_df.to_csv(folder / f"pred_{model_name}_{scenario_name}.csv", index=False)

    tf.keras.backend.clear_session()

    return {
        "Ticker": ticker,
        "Scenario": scenario_name,
        "Model": model_name,
        "ReturnScale": return_scale,
        **{f"Tr_{key}": value for key, value in m_tr.items()},
        **{f"Va_{key}": value for key, value in m_va.items()},
        **{f"Te_{key}": value for key, value in m_te.items()},
        **{f"Tr_{key}": value for key, value in r_tr.items()},
        **{f"Va_{key}": value for key, value in r_va.items()},
        **{f"Te_{key}": value for key, value in r_te.items()},
        **naive,
        **anti_lazy,
        **audit,
    }


def rebuild_metrics(
    results_dir: str,
    tickers: list[str] | None,
    use_mc_dropout: bool,
    write_predictions: bool,
) -> pd.DataFrame:
    results_path = Path(results_dir)
    scenarios = _scenario_by_name()
    rows = []
    failures = []

    ticker_dirs = [
        path
        for path in sorted(results_path.iterdir())
        if path.is_dir() and (tickers is None or path.name in tickers)
    ]

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        for scenario_name, scenario in scenarios.items():
            folder = ticker_dir / scenario_name
            if not folder.is_dir():
                failures.append((ticker, scenario_name, "missing scenario folder"))
                continue

            try:
                windows = _build_windows_from_saved_artifacts(folder, scenario)
            except Exception as exc:
                failures.append((ticker, scenario_name, f"window error: {exc}"))
                continue

            for model_name in MODEL_NAMES:
                model_path = folder / f"model_{model_name}_{scenario_name}.keras"
                if not model_path.exists():
                    failures.append((ticker, scenario_name, f"missing {model_name} model"))
                    continue

                try:
                    row = _evaluate_model(
                        model_path=model_path,
                        ticker=ticker,
                        scenario=scenario,
                        model_name=model_name,
                        windows=windows,
                        use_mc_dropout=use_mc_dropout,
                        write_predictions=write_predictions,
                        folder=folder,
                    )
                    rows.append(row)
                    _log(
                        f"OK {ticker} {scenario_name} {model_name}: "
                        f"Te_MAPE={row['Te_MAPE']:.4f} "
                        f"MoveRatio={row.get('MoveRatio', 0):.4f} "
                        f"CopyRatio={row.get('CopyRatio', 0):.4f}"
                    )
                except Exception as exc:
                    failures.append((ticker, scenario_name, f"{model_name}: {exc}"))

    if not rows:
        raise RuntimeError("No metrics rows were rebuilt.")

    df = pd.DataFrame(rows)
    out_path = results_path / "merged_metrics_ALL.csv"
    df.to_csv(out_path, index=False)

    for ticker, group in df.groupby("Ticker"):
        ticker_path = results_path / ticker
        ticker_path.mkdir(parents=True, exist_ok=True)
        group.to_csv(ticker_path / f"metrics_{ticker}.csv", index=False)

    if "Model" in df.columns and "Scenario" in df.columns:
        summary_rows = []
        for (model, scenario), group in df.groupby(["Model", "Scenario"]):
            total = len(group)
            summary_rows.append(
                {
                    "Model": model,
                    "Scenario": scenario,
                    "N_Total": total,
                    "Avg_MoveRatio": round(group["MoveRatio"].mean(), 4)
                    if "MoveRatio" in group
                    else None,
                    "Avg_CopyRatio": round(group["CopyRatio"].mean(), 4)
                    if "CopyRatio" in group
                    else None,
                    "Avg_DA_test": round(group["DA_test"].mean(), 2)
                    if "DA_test" in group
                    else None,
                    "Avg_NaiveImprove_MAPE": round(
                        group["NaiveImprove_MAPE"].mean(), 4
                    )
                    if "NaiveImprove_MAPE" in group
                    else None,
                    "Pct_BeatNaiveMAPE": round(
                        group["Beats_Naive_MAPE"].mean() * 100, 1
                    )
                    if "Beats_Naive_MAPE" in group
                    else None,
                    "Avg_TrainTest_R2_gap": round(
                        group["TrainTest_R2_gap"].mean(), 4
                    )
                    if "TrainTest_R2_gap" in group
                    else None,
                    "Avg_ValTest_R2_gap": round(group["ValTest_R2_gap"].mean(), 4)
                    if "ValTest_R2_gap" in group
                    else None,
                    "Pct_OverfitSignal": round(group["OverfitSignal"].mean() * 100, 1)
                    if "OverfitSignal" in group
                    else None,
                    "Pct_LazySignal": round(group["LazySignal"].mean() * 100, 1)
                    if "LazySignal" in group
                    else None,
                    "Pct_CopySignal": round(group["CopySignal"].mean() * 100, 1)
                    if "CopySignal" in group
                    else None,
                    "Pct_DirectionSignal": round(
                        group["DirectionSignal"].mean() * 100, 1
                    )
                    if "DirectionSignal" in group
                    else None,
                }
            )
        pd.DataFrame(summary_rows).sort_values(["Model", "Scenario"]).to_csv(
            results_path / "visual_diagnostics_summary.csv", index=False
        )

    if failures:
        fail_path = results_path / "rebuild_metrics_failures.csv"
        pd.DataFrame(failures, columns=["Ticker", "Scenario", "Error"]).to_csv(
            fail_path, index=False
        )
        _log(f"Failures saved to {fail_path}")

    _log(f"Rebuilt {len(df)} rows -> {out_path}")
    _log(f"Tickers: {df['Ticker'].nunique()}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild train/validation/test metrics from saved VN30 models."
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional ticker list. Omit to rebuild all ticker folders.",
    )
    parser.add_argument(
        "--mc-dropout",
        action="store_true",
        help="Use Monte Carlo dropout like main.py. Default is deterministic predict().",
    )
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Overwrite pred_*.csv files using rebuilt deterministic/MC predictions.",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    rebuild_metrics(
        results_dir=args.results_dir,
        tickers=args.tickers,
        use_mc_dropout=args.mc_dropout,
        write_predictions=args.write_predictions,
    )


if __name__ == "__main__":
    main_cli()
