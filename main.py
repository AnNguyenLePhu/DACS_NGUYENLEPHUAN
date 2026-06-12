# =============================================================================
# main.py
# Pipeline dự đoán giá đóng cửa VN30 — chia thành 12 hàm step rõ ràng.
#
# STEP 01  step01_load_validate()       — Đọc CSV, kiểm tra cơ bản
# STEP 02  step02_handle_missing()      — Xử lý missing (chỉ ffill, không bfill)
# STEP 03  step03_feature_engineering() — Tính chỉ báo kỹ thuật (per-ticker)
# STEP 04  step04_clean_features()      — Xóa NaN sau FE, kiểm tra leakage
# STEP 05  step05_split()               — Chronological split 70/15/15
# STEP 06  step06_save_pretrain()       — Lưu data/artifacts trước khi train
# STEP 07  step07_scale()               — Fit scaler CHỈ trên train
# STEP 08  step08_sliding_windows()     — Tạo X/Y windows 3 kịch bản
# STEP 09  step09_train()               — Train 6 model (anti-overfit callbacks)
# STEP 10  step10_predict_evaluate()    — Dự báo, inverse transform, metrics
# STEP 11  step11_overfitting_audit()   — Kiểm tra overfit/lazy/copy giá
# STEP 12  step12_ensemble_and_save()   — Ensemble + lưu kết quả tổng hợp
# =============================================================================

import os
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, models, regularizers, callbacks as keras_callbacks

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import (
    DATA_FILE, RESULTS_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    # Strict Common Dates
    USE_STRICT_COMMON_DATES, EXPECTED_TICKER_COUNT,
    COMMON_DATE_REPORT_PATH, SPLIT_SUMMARY_PATH,
    FEATURE_COLS, N_FEATURES,
    SCENARIOS, MODEL_NAMES,
    EPOCHS, BATCH_SIZE, INITIAL_LR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MONITOR,
    REDUCE_LR_MONITOR, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, REDUCE_LR_MIN_LR,
    H3_EARLY_STOPPING_PATIENCE, H3_REDUCE_LR_PATIENCE, H3_WARMUP_EPOCHS,
    H7_EARLY_STOPPING_PATIENCE, H7_REDUCE_LR_PATIENCE,
    DROPOUT_RATE, L2_LAMBDA,
    PARHYBRID_L2_OVERRIDE, PARHYBRID_SPATIAL_DROP, PARHYBRID_GRAD_CLIP,
    LABEL_SMOOTH_ALPHA, AUG_NOISE_STD_X, AUG_NOISE_STD_Y, AUG_PROB, MC_DROPOUT_SAMPLES,
    DIR_LOSS_WEIGHT_H1, DIR_LOSS_WEIGHT_H3, DIR_LOSS_WEIGHT_H7, USE_DIR_LOSS,
    OVERFIT_R2_GAP_THRESHOLD, DA_PASS_THRESHOLD, DA_WARN_THRESHOLD,
    VOLRATIO_TARGET_MIN, VOLRATIO_TARGET_MAX,
    LAZY_PASS_THRESHOLD, MAPE_MIN_THRESHOLD, MAPE_MAX_THRESHOLD,
    MIN_TICKER_ROWS, USE_TEST_MODE, TEST_TICKERS, SEED,
    USE_SAMPLE_WEIGHTS, SAMPLE_WEIGHT_GAMMA, SAMPLE_WEIGHT_MULTIPLIER, SAMPLE_WEIGHT_CLIP_MAX,
    HIGH_VOL_THRESHOLD, HIGH_VOL_WEIGHT,
    USE_HUBER_LOSS, HUBER_DELTA,
    MAG_CONSTRAINT_RATIO, MAG_CONSTRAINT_WEIGHT,
    MIN_PRED_MOVE_STD, ZERO_MOVE_PENALTY_WEIGHT, RETURN_CORR_LOSS_WEIGHT,
    PASS_REQUIRE_BEATS_NAIVE, PASS_MIN_DA, PASS_REQUIRE_MAPE_BEAT, PASS_MAX_COPY_RATIO,
    MIN_MOVE_RATIO,
    # [FIX] Hằng số mới cho CopyRatio thực sự và lazy/DA thresholds
    COPY_PRICE_REL_THRESHOLD, MAX_LAZY_RATIO, MIN_DA_PASS,
    USE_RETURN_CALIBRATION, RETURN_SCALE_CANDIDATES,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# STEP 01 — LOAD & VALIDATE
# =============================================================================

def step01_load_validate(filepath: str) -> pd.DataFrame:
    """
    Đọc CSV gốc, kiểm tra cơ bản: shape, dtypes, duplicates, sort.
    KHÔNG xử lý missing ở bước này.
    """
    _banner("STEP 01 — LOAD & VALIDATE")

    df = pd.read_csv(filepath)
    _log(f"File         : '{filepath}'")
    _log(f"Shape (raw)  : {df.shape}")

    # Parse ngày
    df["TradingDate"] = pd.to_datetime(df["TradingDate"], errors="coerce")
    bad_dates = df["TradingDate"].isna().sum()
    if bad_dates:
        _warn(f"{bad_dates} rows có TradingDate không hợp lệ → drop")
        df = df.dropna(subset=["TradingDate"])

    # Sort — BẮT BUỘC trước mọi thao tác
    df = df.sort_values(["Ticker", "TradingDate"]).reset_index(drop=True)

    # Drop duplicates
    n_dup = df.duplicated(subset=["Ticker", "TradingDate"]).sum()
    if n_dup:
        _warn(f"{n_dup} dòng duplicate (Ticker, Date) → drop")
        df = df.drop_duplicates(subset=["Ticker", "TradingDate"]).reset_index(drop=True)

    # Drop Close <= 0 (lỗi dữ liệu: VIB 2018-07-23)
    bad_close = df["Close"] <= 0
    if bad_close.any():
        _warn(f"Drop {bad_close.sum()} dòng Close<=0: "
              f"{df.loc[bad_close, ['Ticker','TradingDate','Close']].values.tolist()}")
        df = df[~bad_close].reset_index(drop=True)

    tickers = sorted(df["Ticker"].unique().tolist())
    _log(f"Tickers      : {len(tickers)} — {tickers}")
    _log(f"Rows         : {len(df):,}")
    _log(f"Date range   : {df['TradingDate'].min().date()} → {df['TradingDate'].max().date()}")
    _log(f"Missing      : {df[['Open','High','Low','Close','Volume','VN30_Close','VN30_Volume']].isna().sum().to_dict()}")
    return df


# =============================================================================
# STEP 02 — HANDLE MISSING
# =============================================================================

def step02_handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý NaN cho các cột OHLCV và VN30.
    Chỉ ffill (forward fill) — TUYỆT ĐỐI KHÔNG bfill.
    bfill dùng dữ liệu tương lai để lấp quá khứ → data leakage.
    Drop nếu đầu chuỗi vẫn còn NaN (chưa có giá trị trước để ffill).
    """
    _banner("STEP 02 — HANDLE MISSING")

    fill_cols = ["Open", "High", "Low", "Close", "Volume", "VN30_Close", "VN30_Volume"]
    before = df[fill_cols].isna().sum()
    _log(f"Missing trước ffill: {before[before > 0].to_dict()}")

    # ffill theo từng ticker — KHÔNG bfill
    df[fill_cols] = df.groupby("Ticker")[fill_cols].transform(lambda g: g.ffill())

    still_na = df[fill_cols].isna().sum().sum()
    if still_na:
        _warn(f"Còn {still_na} NaN sau ffill (đầu chuỗi) → drop")
        df = df.dropna(subset=fill_cols).reset_index(drop=True)

    after = df[fill_cols].isna().sum().sum()
    _log(f"Missing sau xử lý: {after}  ✓")
    _log(f"Shape sau clean  : {df.shape}")

    assert after == 0, "Còn NaN sau step02!"
    return df


# =============================================================================
# STEP 03 — FEATURE ENGINEERING (per-ticker)
# =============================================================================

def _fe_one_ticker(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Tính chỉ báo kỹ thuật cho 1 ticker.
    Chỉ dùng shift/rolling/ewm theo chiều quá khứ — KHÔNG look-ahead.
    """
    t   = df_t.copy().reset_index(drop=True)
    c   = t["Close"]
    h   = t["High"]
    lo  = t["Low"]
    o   = t["Open"]
    vol = t["Volume"].clip(lower=0)
    eps = 1e-9

    # ── Returns ──────────────────────────────────────────────────────────────
    t["return_1d"]    = c.pct_change()
    t["log_return_1d"] = np.log((c / (c.shift(1) + eps)).clip(lower=eps))

    t["return_3d"] = c.pct_change(3)
    t["return_5d"] = c.pct_change(5)

    t["volatility_5d"] = t["log_return_1d"].rolling(5).std()
    t["volatility_10d"] = t["log_return_1d"].rolling(10).std()

    # ── Spread nội ngày ──────────────────────────────────────────────────────
    t["HL_Range"]  = h - lo
    t["OC_Change"] = c - o
    t["open_close_ratio"] = o / (c + eps) - 1
    t["high_close_ratio"] = h / (c + eps) - 1
    t["low_close_ratio"] = lo / (c + eps) - 1
    t["hl_range_ratio"] = (h - lo) / (c + eps)
    t["oc_change_ratio"] = (c - o) / (c + eps)

    # ── Volume ───────────────────────────────────────────────────────────────
    t["Volume_Change"] = np.log((vol / (vol.shift(1) + eps)).clip(lower=eps))

    # ── Moving Averages ──────────────────────────────────────────────────────
    t["MA5"]  = c.rolling(5).mean()
    t["MA10"] = c.rolling(10).mean()
    t["MA20"] = c.rolling(20).mean()

    volume_ma20 = vol.rolling(20).mean()
    t["volume_ma_ratio"] = vol / (volume_ma20 + eps)

    t["price_ma5_gap"] = c / (t["MA5"] + eps) - 1
    t["price_ma20_gap"] = c / (t["MA20"] + eps) - 1
    t["ma5_close_ratio"] = t["MA5"] / (c + eps) - 1
    t["ma10_close_ratio"] = t["MA10"] / (c + eps) - 1
    t["ma20_close_ratio"] = t["MA20"] / (c + eps) - 1

    # ── EMA (log-return để stationary) ───────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    t["EMA12"] = np.log((ema12 / (ema12.shift(1) + eps)).clip(lower=eps))
    t["EMA26"] = np.log((ema26 / (ema26.shift(1) + eps)).clip(lower=eps))

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    delta   = c.diff()
    avg_g   = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_l   = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    t["RSI14"] = 100.0 - (100.0 / (1.0 + avg_g / (avg_l + eps)))

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_line      = ema12 - ema26
    macd_signal    = macd_line.ewm(span=9, adjust=False).mean()
    t["MACD"]        = macd_line
    t["MACD_Signal"] = macd_signal

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────────────────
    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    t["BB_Upper"] = bb_mid + 2 * bb_std
    t["BB_Lower"] = bb_mid - 2 * bb_std
    t["bb_upper_close_ratio"] = t["BB_Upper"] / (c + eps) - 1
    t["bb_lower_close_ratio"] = t["BB_Lower"] / (c + eps) - 1

    # ── ATR(14) ───────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    t["ATR14"] = tr.rolling(14).mean()
    t["atr14_close_ratio"] = t["ATR14"] / (c + eps)

    # ── Gap_Flag (khoảng trống giao dịch) ────────────────────────────────────
    days_gap      = t["TradingDate"].diff().dt.days.fillna(1)
    t["Gap_Flag"] = (days_gap > 5).astype(int)

    # ── [NEW] Breakout features ──────────────────────────────────────────
    # Rolling High/Low 20 phiên (shift(1): chỉ dùng dữ liệu quá khứ, không look-ahead)
    t["rolling_max_20"] = c.rolling(20).max()
    t["rolling_min_20"] = c.rolling(20).min()
    t["rolling_max20_close_ratio"] = t["rolling_max_20"] / (c + eps) - 1
    t["rolling_min20_close_ratio"] = t["rolling_min_20"] / (c + eps) - 1

    t["breakout_up"] = (
        c > t["rolling_max_20"].shift(1)
    ).astype(int)

    t["breakout_down"] = (
        c < t["rolling_min_20"].shift(1)
    ).astype(int)

    volume_ma20 = vol.shift(1).rolling(20).mean()
    t["volume_spike"]  = (vol > 2.0 * (volume_ma20 + eps)).astype(np.float32)
    return t


def step03_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Chạy FE cho tất cả tickers. Trả về dict {ticker: DataFrame}.
    """
    _banner("STEP 03 — FEATURE ENGINEERING")
    result = {}
    for ticker in sorted(df["Ticker"].unique()):
        df_t  = df[df["Ticker"] == ticker].copy()
        df_fe = _fe_one_ticker(df_t)
        result[ticker] = df_fe
        _log(f"[{ticker}] {len(df_fe):,} rows | indicators computed")
    _log(f"Tổng: {len(result)} tickers")
    return result


# =============================================================================
# STEP 04 — CLEAN FEATURES (dropna, validate, no bfill)
# =============================================================================

def step04_clean_features(feat_dict: dict) -> dict:
    """
    Xóa NaN sau FE bằng dropna() — KHÔNG ffill/bfill trên FEATURE_COLS.
    Kiểm tra: không NaN, không Inf, đủ MIN_TICKER_ROWS.
    Trả về dict {ticker: DataFrame} (loại ticker không đủ dữ liệu).
    """
    _banner("STEP 04 — CLEAN FEATURES")
    clean = {}
    for ticker, df_fe in feat_dict.items():
        # dropna CHỈ trên FEATURE_COLS — không fill
        df_c = df_fe.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        # Kiểm tra Inf
        inf_count = np.isinf(df_c[FEATURE_COLS].values.astype(float)).sum()
        if inf_count:
            _warn(f"[{ticker}] {inf_count} Inf values → drop")
            df_c = df_c.replace([np.inf, -np.inf], np.nan)
            df_c = df_c.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        if len(df_c) < MIN_TICKER_ROWS:
            _warn(f"[{ticker}] Bỏ qua — chỉ {len(df_c)} rows sau dropna (cần ≥ {MIN_TICKER_ROWS})")
            continue

        # Xác nhận không còn NaN/Inf
        assert df_c[FEATURE_COLS].isna().sum().sum() == 0, f"[{ticker}] còn NaN!"
        assert not np.isinf(df_c[FEATURE_COLS].values.astype(float)).any(), f"[{ticker}] còn Inf!"

        clean[ticker] = df_c
        _log(f"[{ticker}] {len(df_c):,} rows | {N_FEATURES} features | OK")

    _log(f"Tổng hợp lệ: {len(clean)}/{len(feat_dict)} tickers")
    return clean


# =============================================================================
# STRICT COMMON DATE FILTER  (Option A)
# =============================================================================

def keep_common_trading_dates(
    df: pd.DataFrame,
    date_col: str = "TradingDate",
    ticker_col: str = "Ticker",
    expected_tickers: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only trading dates where all expected tickers have data.
    This ensures all VN30 tickers are aligned on the same calendar.

    Returns:
        filtered      — DataFrame with only common-date rows
        ticker_count  — per-date ticker count (used for the report)
    """
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    ticker_count = (
        temp.groupby(date_col)[ticker_col]
        .nunique()
        .reset_index(name="n_tickers")
        .sort_values(date_col)
    )

    common_dates = ticker_count.loc[
        ticker_count["n_tickers"] == expected_tickers,
        date_col,
    ]

    filtered = temp[temp[date_col].isin(common_dates)].copy()

    print("\n========== STRICT COMMON DATE FILTER ==========")
    print(f"  Expected tickers per date  : {expected_tickers}")
    print(f"  Original rows              : {len(temp):,}")
    print(f"  Filtered rows              : {len(filtered):,}")
    print(f"  Original trading dates     : {temp[date_col].nunique():,}")
    print(f"  Common trading dates kept  : {filtered[date_col].nunique():,}")
    print(f"  Removed trading dates      : "
          f"{temp[date_col].nunique() - filtered[date_col].nunique():,}")

    return filtered, ticker_count


def get_global_split_dates(
    df: pd.DataFrame,
    date_col: str = "TradingDate",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    """
    Create global chronological split dates based on the common trading calendar.

    Returns:
        (train_end_date, val_end_date, test_start_date)
    """
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])
    all_dates = sorted(temp[date_col].dropna().unique())

    if len(all_dates) < 10:
        raise ValueError(
            "Not enough common trading dates after strict filtering. "
            f"Got {len(all_dates)} dates — need at least 10."
        )

    n_dates      = len(all_dates)
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx   = int(n_dates * (train_ratio + val_ratio))

    if train_end_idx <= 0:
        raise ValueError("train_end_idx <= 0. Check TRAIN_RATIO.")
    if val_end_idx <= train_end_idx:
        raise ValueError("val_end_idx <= train_end_idx. Check VAL_RATIO.")
    if val_end_idx >= n_dates:
        raise ValueError(
            "val_end_idx >= n_dates — test split would be empty. "
            "Reduce TRAIN_RATIO + VAL_RATIO so that TEST_RATIO > 0."
        )

    train_end_date  = all_dates[train_end_idx - 1]
    val_end_date    = all_dates[val_end_idx - 1]
    test_start_date = all_dates[val_end_idx]

    print("\n========== GLOBAL DATE SPLIT ==========")
    print(f"  Total common trading dates : {n_dates:,}")
    print(f"  Train  dates               : {train_end_idx}  "
          f"(up to {pd.Timestamp(train_end_date).date()})")
    print(f"  Val    dates               : {val_end_idx - train_end_idx}  "
          f"({pd.Timestamp(all_dates[train_end_idx]).date()} → "
          f"{pd.Timestamp(val_end_date).date()})")
    print(f"  Test   dates               : {n_dates - val_end_idx}  "
          f"(from {pd.Timestamp(test_start_date).date()})")
    print(f"  Train end date             : {pd.Timestamp(train_end_date).date()}")
    print(f"  Validation end date        : {pd.Timestamp(val_end_date).date()}")
    print(f"  Test start date            : {pd.Timestamp(test_start_date).date()}")

    return train_end_date, val_end_date, test_start_date


def apply_global_date_split(
    df: pd.DataFrame,
    train_end_date,
    val_end_date,
    date_col: str = "TradingDate",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the same train/validation/test date boundaries to all tickers.
    Scalers must be fitted ONLY on train_df to prevent data leakage.
    """
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    train_df = temp[temp[date_col] <= train_end_date].copy()
    val_df   = temp[
        (temp[date_col] > train_end_date) &
        (temp[date_col] <= val_end_date)
    ].copy()
    test_df  = temp[temp[date_col] > val_end_date].copy()

    return train_df, val_df, test_df


def save_global_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str = "results/global_split_summary.csv",
    date_col: str = "TradingDate",
    ticker_col: str = "Ticker",
) -> pd.DataFrame:
    """
    Save a summary proving that all tickers share the same split periods.
    """
    rows = []
    for split_name, part in [
        ("train",      train_df),
        ("validation", val_df),
        ("test",       test_df),
    ]:
        if part.empty:
            raise ValueError(f"{split_name} split is empty — cannot save summary.")

        summary = (
            part.groupby(ticker_col)[date_col]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        for _, row in summary.iterrows():
            rows.append({
                "Split":     split_name,
                "Ticker":    row[ticker_col],
                "StartDate": row["min"],
                "EndDate":   row["max"],
                "Rows":      row["count"],
            })

        print(f"\n========== {split_name.upper()} SPLIT SUMMARY ==========")
        print(f"  Rows    : {len(part):,}")
        print(f"  Tickers : {part[ticker_col].nunique()}")
        print(f"  Date range: {part[date_col].min().date()} → "
              f"{part[date_col].max().date()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved global split summary → {output_path}")
    return report


def _validate_global_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expected_tickers: int = 30,
    date_col: str = "TradingDate",
    ticker_col: str = "Ticker",
) -> None:
    """
    Sanity checks after strict common-date filter and global split.
    Raises ValueError with a clear message if any check fails.
    """
    _banner("GLOBAL SPLIT VALIDATION")

    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        n_t = part[ticker_col].nunique()
        if n_t != expected_tickers:
            raise ValueError(
                f"[VALIDATION FAIL] {name} split has {n_t} tickers "
                f"— expected {expected_tickers}."
            )
        _log(f"[{name:10s}] tickers={n_t} ✓")

    # All tickers share the same date boundaries within each split
    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        starts = part.groupby(ticker_col)[date_col].min()
        ends   = part.groupby(ticker_col)[date_col].max()
        if starts.nunique() != 1:
            raise ValueError(
                f"[VALIDATION FAIL] {name} split — tickers have different start dates:\n"
                f"{starts.value_counts()}"
            )
        if ends.nunique() != 1:
            raise ValueError(
                f"[VALIDATION FAIL] {name} split — tickers have different end dates:\n"
                f"{ends.value_counts()}"
            )
        _log(f"[{name:10s}] all tickers share same start/end dates ✓")

    # No overlap between splits
    tr_dates = set(train_df[date_col].unique())
    va_dates = set(val_df[date_col].unique())
    te_dates = set(test_df[date_col].unique())

    if tr_dates & va_dates:
        raise ValueError(
            f"[VALIDATION FAIL] Overlap between train and validation: "
            f"{sorted(tr_dates & va_dates)[:5]} ..."
        )
    if va_dates & te_dates:
        raise ValueError(
            f"[VALIDATION FAIL] Overlap between validation and test: "
            f"{sorted(va_dates & te_dates)[:5]} ..."
        )
    if tr_dates & te_dates:
        raise ValueError(
            f"[VALIDATION FAIL] Overlap between train and test: "
            f"{sorted(tr_dates & te_dates)[:5]} ..."
        )
    _log("No date overlap between splits ✓")

    # Chronological order: train_end < val_start < test_start
    tr_end   = train_df[date_col].max()
    va_start = val_df[date_col].min()
    va_end   = val_df[date_col].max()
    te_start = test_df[date_col].min()

    if not (tr_end < va_start):
        raise ValueError(
            f"[VALIDATION FAIL] train_end ({tr_end.date()}) is not < "
            f"val_start ({va_start.date()})."
        )
    if not (va_end < te_start):
        raise ValueError(
            f"[VALIDATION FAIL] val_end ({va_end.date()}) is not < "
            f"test_start ({te_start.date()})."
        )
    _log(f"Chronological order: train_end={tr_end.date()} < "
         f"val_start={va_start.date()} < test_start={te_start.date()} ✓")

    _log("All validation checks PASSED ✓")


# =============================================================================
# STEP 05 — CHRONOLOGICAL SPLIT 70/15/15 (per-ticker slice of global split)
# =============================================================================

def step05_split(
    df_ticker: pd.DataFrame,
    train_end_date=None,
    val_end_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Chia theo thứ tự thời gian — KHÔNG shuffle, KHÔNG random_state.

    Khi USE_STRICT_COMMON_DATES=True, nhận train_end_date / val_end_date từ
    get_global_split_dates() và slice ticker theo ranh giới ngày chung đó.
    Khi False (legacy), tự tính theo tỉ lệ riêng của ticker.

    Returns (df_train, df_val, df_test, split_info).
    """
    if train_end_date is not None and val_end_date is not None:
        # ── Global date split (Option A) ─────────────────────────────────────
        date_col = "TradingDate"
        df_ticker = df_ticker.copy()
        df_ticker[date_col] = pd.to_datetime(df_ticker[date_col])

        df_tr = df_ticker[df_ticker[date_col] <= train_end_date].copy()
        df_va = df_ticker[
            (df_ticker[date_col] > train_end_date) &
            (df_ticker[date_col] <= val_end_date)
        ].copy()
        df_te = df_ticker[df_ticker[date_col] > val_end_date].copy()
    else:
        # ── Legacy per-ticker split ───────────────────────────────────────────
        n    = len(df_ticker)
        n_tr = int(n * TRAIN_RATIO)
        n_va = int(n * VAL_RATIO)

        df_tr = df_ticker.iloc[:n_tr].copy()
        df_va = df_ticker.iloc[n_tr : n_tr + n_va].copy()
        df_te = df_ticker.iloc[n_tr + n_va:].copy()

    if df_tr.empty or df_va.empty or df_te.empty:
        raise ValueError(
            f"step05_split produced an empty split for ticker "
            f"'{df_ticker['Ticker'].iloc[0] if len(df_ticker) > 0 else '?'}'. "
            "Check global split dates or ticker row count."
        )

    info = {
        "n_total":    len(df_ticker),
        "n_train":    len(df_tr),
        "n_val":      len(df_va),
        "n_test":     len(df_te),
        "train_start": str(df_tr["TradingDate"].iloc[0].date()),
        "train_end":   str(df_tr["TradingDate"].iloc[-1].date()),
        "val_start":   str(df_va["TradingDate"].iloc[0].date()),
        "val_end":     str(df_va["TradingDate"].iloc[-1].date()),
        "test_start":  str(df_te["TradingDate"].iloc[0].date()),
        "test_end":    str(df_te["TradingDate"].iloc[-1].date()),
    }
    _log(f"  Train {len(df_tr):,}  ({info['train_start']} → {info['train_end']})")
    _log(f"  Val   {len(df_va):,}  ({info['val_start']}   → {info['val_end']})")
    _log(f"  Test  {len(df_te):,}  ({info['test_start']}  → {info['test_end']})")
    return df_tr, df_va, df_te, info



# =============================================================================
# STEP 06 — SAVE PRE-TRAIN ARTIFACTS
# =============================================================================

def step06_save_pretrain(
    ticker: str,
    scenario: dict,
    df_full: pd.DataFrame,
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    split_info: dict,
    out_root: str,
) -> str:
    """
    Lưu data trước khi train:
      {out_root}/{ticker}/{scenario}/
        ├── processed_full.csv
        ├── train.csv / val.csv / test.csv
        ├── split_info.json
        └── feature_list.json
    Scaler (pkl) được lưu sau khi fit ở step07.
    Returns thư mục đã lưu.
    """
    folder = os.path.join(out_root, ticker, scenario["name"])
    os.makedirs(folder, exist_ok=True)

    df_full.to_csv(os.path.join(folder, "processed_full.csv"), index=False)
    df_tr.to_csv(  os.path.join(folder, "train.csv"),          index=False)
    df_va.to_csv(  os.path.join(folder, "val.csv"),            index=False)
    df_te.to_csv(  os.path.join(folder, "test.csv"),           index=False)

    with open(os.path.join(folder, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    with open(os.path.join(folder, "feature_list.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "n_features": N_FEATURES}, f, indent=2)
    
    metadata = {
    "ticker": ticker,
    "scenario": scenario["name"],
    "lookback": scenario["lookback"],
    "horizon": scenario["horizon"],
    "n_features": N_FEATURES,
    "feature_cols": FEATURE_COLS,
    "full_rows": len(df_full),
    "train_rows": len(df_tr),
    "val_rows": len(df_va),
    "test_rows": len(df_te),
    "target": f"Future_Log_Return_{scenario['horizon']}d",
    "target_formula": "log(Future_Close_h / Current_Close)",
    "prediction_formula": "Pred_Close = Base_Close * exp(Pred_Log_Return)"
    }

    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    target_info = {
        "target_type": "future_log_return",
        "horizon": scenario["horizon"],
        "target_column": f"Future_Log_Return_{scenario['horizon']}d",
        "future_close_column": f"Future_Close_{scenario['horizon']}d",
        "base_close_column": "Close",
        "formula": "Future_Log_Return_h = log(Future_Close_h / Close)",
        "inverse_formula": "Pred_Close = Base_Close * exp(Pred_Log_Return)"
}

    with open(os.path.join(folder, "target_info.json"), "w", encoding="utf-8") as f:
        json.dump(target_info, f, indent=2, ensure_ascii=False)
    _log(f"[{ticker}|{scenario['name']}] Pre-train artifacts → {folder}")
    return folder


# =============================================================================
# STEP 07 — SCALE (fit CHỈ trên train)
# =============================================================================

def step07_scale(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    folder: str,
) -> tuple:
    """
    MinMaxScaler(X): fit CHỈ trên train, transform val+test.
    Lưu scaler_x.pkl và scaler_y.pkl vào folder.
    target_scaler (StandardScaler cho Y) được fit ở step08 sau khi tạo windows.

    Returns (scaler_x, tr_scaled, va_scaled, te_scaled,
             tr_dates, va_dates, te_dates,
             raw_close_tr, raw_close_va, raw_close_te)
    """
    scaler_x   = MinMaxScaler(feature_range=(0, 1))
    tr_scaled  = scaler_x.fit_transform(df_tr[FEATURE_COLS].values)   # fit CHỈ train
    va_scaled  = scaler_x.transform(df_va[FEATURE_COLS].values)       # transform bằng scaler train
    te_scaled  = scaler_x.transform(df_te[FEATURE_COLS].values)       # transform bằng scaler train

    joblib.dump(scaler_x, os.path.join(folder, "scaler_x.pkl"))
    _log(f"  scaler_x fit on train ({len(df_tr)} rows) — saved scaler_x.pkl")

    return (
        scaler_x,
        tr_scaled, va_scaled, te_scaled,
        df_tr["TradingDate"].values, df_va["TradingDate"].values, df_te["TradingDate"].values,
        df_tr["Close"].values.astype(np.float64),
        df_va["Close"].values.astype(np.float64),
        df_te["Close"].values.astype(np.float64),
    )


# =============================================================================
# STEP 08 — SLIDING WINDOWS
# =============================================================================

def _make_windows(
    scaled: np.ndarray,
    raw_close: np.ndarray,
    dates: np.ndarray,
    lookback: int,
    horizon: int,
    context_scaled: np.ndarray = None,
    context_close:  np.ndarray = None,
    context_dates:  np.ndarray = None,
) -> tuple:
    """
    X[i]      = scaled[i-lookback : i]   shape (lookback, N_FEATURES)
    Y[i]      = log_return tại [i..i+horizon-1]  shape (horizon,)
    Y_dates   = dates tương ứng với Y
    last_close= Close tại i-1 (để inverse về giá)
    """
    if context_scaled is not None:
        scaled    = np.concatenate([context_scaled, scaled], axis=0)
        raw_close = np.concatenate([context_close,  raw_close], axis=0)
        dates     = np.concatenate([context_dates,  dates], axis=0)

    X, Y, Y_dates, last_close = [], [], [], []
    total = len(scaled)
    for i in range(lookback, total - horizon + 1):
        close_w = raw_close[i - 1 : i + horizon]          # Close[t-1], Close[t], ..., Close[t+h-1]
        log_ret = np.log(close_w[1:] / (close_w[:-1] + 1e-9))
        X.append(scaled[i - lookback : i])
        Y.append(log_ret)
        Y_dates.append(dates[i : i + horizon])
        last_close.append(close_w[0])                      # Close tại t-1

    if not X:
        return None, None, None, None
    return (
        np.array(X,          dtype=np.float32),
        np.array(Y,          dtype=np.float32),
        np.array(Y_dates,    dtype=object),
        np.array(last_close, dtype=np.float64),
    )


def step08_sliding_windows(
    tr_scaled: np.ndarray,  va_scaled: np.ndarray,  te_scaled: np.ndarray,
    raw_close_tr: np.ndarray, raw_close_va: np.ndarray, raw_close_te: np.ndarray,
    tr_dates: np.ndarray,   va_dates: np.ndarray,   te_dates: np.ndarray,
    scenario: dict,
    folder: str,
) -> tuple | None:
    """
    Tạo sliding windows cho 3 tập. Fit StandardScaler trên Y_train.
    Lưu scaler_y.pkl. Returns None nếu không đủ windows.
    """
    lb = scenario["lookback"]
    h  = scenario["horizon"]

    X_tr, Y_tr, Yd_tr, Lc_tr = _make_windows(tr_scaled, raw_close_tr, tr_dates, lb, h)
    X_va, Y_va, Yd_va, Lc_va = _make_windows(
        va_scaled, raw_close_va, va_dates, lb, h,
        context_scaled=tr_scaled[-lb:],
        context_close=raw_close_tr[-lb:],
        context_dates=tr_dates[-lb:],
    )
    X_te, Y_te, Yd_te, Lc_te = _make_windows(
        te_scaled, raw_close_te, te_dates, lb, h,
        context_scaled=va_scaled[-lb:],
        context_close=raw_close_va[-lb:],
        context_dates=va_dates[-lb:],
    )

    if X_tr is None or len(X_tr) < 10 or X_va is None or len(X_va) < 1 or X_te is None or len(X_te) < 1:
        _warn(f"  Không đủ windows (tr={0 if X_tr is None else len(X_tr)}, "
              f"va={0 if X_va is None else len(X_va)}, "
              f"te={0 if X_te is None else len(X_te)}) → bỏ qua")
        return None

    # StandardScaler cho Y — fit CHỈ trên Y_train
    scaler_y  = StandardScaler()
    Y_tr_s    = scaler_y.fit_transform(Y_tr.reshape(-1, 1)).reshape(Y_tr.shape)
    Y_va_s    = scaler_y.transform(Y_va.reshape(-1, 1)).reshape(Y_va.shape)
    # Y_te không scale (chỉ dùng cho inverse)

    joblib.dump(scaler_y, os.path.join(folder, "scaler_y.pkl"))

    _log(f"  Windows: train={len(X_tr):,} | val={len(X_va):,} | test={len(X_te):,} | "
         f"features={X_tr.shape[2]} | horizon={h}")

    return {
        "X_tr": X_tr,  "Y_tr_s": Y_tr_s,  "Y_tr": Y_tr, "Yd_tr": Yd_tr, "Lc_tr": Lc_tr,
        "X_va": X_va,  "Y_va_s": Y_va_s,  "Y_va": Y_va, "Yd_va": Yd_va, "Lc_va": Lc_va,
        "X_te": X_te,                      "Y_te": Y_te, "Yd_te": Yd_te, "Lc_te": Lc_te,
        "scaler_y": scaler_y,
    }


# =============================================================================
# STEP 09 — BUILD & TRAIN MODELS
# =============================================================================

# ── Custom loss factory: Huber + Directional penalty (scale-aware) ────────────
# BUG CŨ: tf.sign(y_true) so sánh với 0 trong không gian scaled.
# Sau StandardScaler, 0 không còn là "không đổi" thực sự.
# FIX: Tính y_zero_scaled = scaler_y.transform([[0.0]])[0,0] rồi so sánh y - y_zero_scaled.
DIR_PENALTY_WEIGHT = 0.30  # giữ lại để tham chiếu; giá trị thực dùng từ config theo horizon

def make_directional_huber_loss(
    y_zero_scaled: float,
    dir_weight: float = 0.30,
    mag_ratio: float = MAG_CONSTRAINT_RATIO,
    mag_weight: float = MAG_CONSTRAINT_WEIGHT,
    min_pred_move: float = MIN_PRED_MOVE_STD,
    zero_move_weight: float = ZERO_MOVE_PENALTY_WEIGHT,
    corr_weight: float = RETURN_CORR_LOSS_WEIGHT,
):
    """
    Tạo loss function biết vị trí 'zero return' trong không gian scaled.

    Args:
        y_zero_scaled: scaler_y.transform([[0.0]])[0, 0]
                       — giá trị 0.0 log-return sau StandardScaler
        dir_weight   : trọng số penalize sai chiều (0.20–0.40)

    Returns:
        loss(y_true, y_pred) → scalar tensor

    Multi-step safe: tf.reduce_mean hoạt động trên mọi shape (batch, h).
    """
    huber_fn = tf.keras.losses.Huber(delta=HUBER_DELTA)
    mse_fn = tf.keras.losses.MeanSquaredError()

    def loss(y_true, y_pred):
        base_loss = huber_fn(y_true, y_pred) if USE_HUBER_LOSS else mse_fn(y_true, y_pred)
        # So sánh với y_zero_scaled thay vì 0 — đây là điểm fix cốt lõi
        true_move = y_true - y_zero_scaled
        pred_move = y_pred - y_zero_scaled
        true_sign = tf.stop_gradient(tf.sign(true_move))
        direction_penalty = tf.reduce_mean(tf.nn.relu(-true_sign * pred_move))
        magnitude_floor = mag_ratio * tf.abs(true_move)
        magnitude_penalty = tf.reduce_mean(tf.nn.relu(magnitude_floor - tf.abs(pred_move)))
        nonzero_weight = tf.clip_by_value(tf.abs(true_move) / (min_pred_move + 1e-6), 0.0, 1.0)
        zero_move_penalty = tf.reduce_mean(
            nonzero_weight * tf.nn.relu(min_pred_move - tf.abs(pred_move))
        )

        true_flat = tf.reshape(true_move, [-1])
        pred_flat = tf.reshape(pred_move, [-1])
        true_centered = true_flat - tf.reduce_mean(true_flat)
        pred_centered = pred_flat - tf.reduce_mean(pred_flat)
        denom = (
            tf.sqrt(tf.reduce_sum(tf.square(true_centered)) + 1e-6)
            * tf.sqrt(tf.reduce_sum(tf.square(pred_centered)) + 1e-6)
        )
        corr = tf.reduce_sum(true_centered * pred_centered) / denom
        corr_penalty = 1.0 - corr
        if not USE_DIR_LOSS:
            direction_penalty = 0.0
        return (
            base_loss
            + dir_weight * direction_penalty
            + mag_weight * magnitude_penalty
            + zero_move_weight * zero_move_penalty
            + corr_weight * corr_penalty
        )

    loss.__name__ = f"dir_huber_loss_w{dir_weight}"
    return loss


def _build_model(name: str, lookback: int, horizon: int) -> tf.keras.Model:
    """Tạo model theo tên. Tất cả dùng padding='causal' cho Conv1D."""
    nf = N_FEATURES
    dr = DROPOUT_RATE
    l2 = L2_LAMBDA
    inp = layers.Input(shape=(lookback, nf))

    if name == "LSTM":
        x = layers.LSTM(48, return_sequences=True, kernel_regularizer=regularizers.l2(l2))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.LSTM(24, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(16, activation="relu")(x)

    elif name == "GRU":
        x = layers.GRU(48, return_sequences=True, kernel_regularizer=regularizers.l2(l2))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.GRU(24, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(16, activation="relu")(x)

    elif name == "RNN":
        x = layers.SimpleRNN(48, return_sequences=True, kernel_regularizer=regularizers.l2(l2))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dr)(x)
        x = layers.SimpleRNN(24, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(16, activation="relu")(x)

    else:
        raise ValueError(f"Unknown model: {name}")


    out = layers.Dense(horizon)(x)
    m   = models.Model(inp, out, name=name)
    # Không compile ở đây — compile với loss đúng (scale-aware) trong step09_train
    return m


def _build_callbacks(horizon: int, ckpt_path: str) -> list:
    """EarlyStopping + ReduceLROnPlateau + ModelCheckpoint."""
    if horizon == 3:
        es_p, lr_p = H3_EARLY_STOPPING_PATIENCE, H3_REDUCE_LR_PATIENCE
    elif horizon == 7:
        es_p, lr_p = H7_EARLY_STOPPING_PATIENCE, H7_REDUCE_LR_PATIENCE
    else:
        es_p, lr_p = EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE

    return [
        keras_callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=es_p,
            restore_best_weights=True,    # ← giữ weights tốt nhất
            verbose=0,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor=REDUCE_LR_MONITOR,
            factor=REDUCE_LR_FACTOR,
            patience=lr_p,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=0,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


def _label_smooth(Y: np.ndarray, alpha: float = LABEL_SMOOTH_ALPHA) -> np.ndarray:
    return (1.0 - alpha) * Y + alpha * float(np.mean(Y))


def _augment(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Gaussian noise injection để chống memorization."""
    mask = np.random.rand(len(X)) < AUG_PROB
    X_aug = X.copy()
    Y_aug = Y.copy()
    X_aug[mask] += np.random.normal(0, AUG_NOISE_STD_X, X_aug[mask].shape).astype(np.float32)
    Y_aug[mask] += np.random.normal(0, AUG_NOISE_STD_Y, Y_aug[mask].shape).astype(np.float32)
    return X_aug, Y_aug


def _mc_predict(model: tf.keras.Model, X: np.ndarray, n: int = MC_DROPOUT_SAMPLES) -> np.ndarray:
    """Monte Carlo dropout inference."""
    preds = np.stack([model(X, training=True).numpy() for _ in range(n)], axis=0)
    return preds.mean(axis=0)


def step09_train(
    w: dict,
    model_name: str,
    scenario: dict,
    folder: str,
) -> dict | None:
    """
    Build, augment, label-smooth, train model với callbacks chống overfit.
    shuffle=False — time-series.
    Returns dict kết quả hoặc None nếu lỗi.
    """
    lb = scenario["lookback"]
    h  = scenario["horizon"]
    sc = scenario["name"]

    tf.keras.backend.clear_session()
    model    = _build_model(model_name, lb, h)
    ckpt_path = os.path.join(folder, f"best_{model_name}_{sc}.keras")
    cbs       = _build_callbacks(h, ckpt_path)

    # ── [FIX 1] Compile với loss factory biết y_zero_scaled ─────────────────
    # scaler_y đã fit trên Y_train (log-return raw) → transform([[0.0]]) cho điểm "0 return"
    y_zero_scaled = float(w["scaler_y"].transform([[0.0]])[0, 0])
    dir_weight_map = {1: DIR_LOSS_WEIGHT_H1, 3: DIR_LOSS_WEIGHT_H3, 7: DIR_LOSS_WEIGHT_H7}
    dir_w = dir_weight_map.get(h, 0.30)
    loss_fn = make_directional_huber_loss(y_zero_scaled=y_zero_scaled, dir_weight=dir_w)

    opt = tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR,
        clipnorm=PARHYBRID_GRAD_CLIP if model_name == "ParHybrid" else 1.0,
    )
    model.compile(optimizer=opt, loss=loss_fn, metrics=["mae"])

    X_tr_aug, Y_tr_aug = _augment(w["X_tr"], w["Y_tr_s"])
    Y_smooth = _label_smooth(Y_tr_aug)

    # ── [FIX 2] Sample weights: dùng Y_tr RAW (log-return gốc), không dùng scaled ──
    # HIGH_VOL_THRESHOLD = 0.015 là ngưỡng log-return thực (không phải scaled).
    # w["Y_tr"] shape (N, h) — dùng max abs trên tất cả horizon steps.
    sw = None
    if USE_SAMPLE_WEIGHTS:
        abs_y_raw = np.max(np.abs(w["Y_tr"]), axis=1)   # (N,) — raw log-return
        median_move = np.median(abs_y_raw) + 1e-9
        scaled_move = np.power(abs_y_raw / median_move, SAMPLE_WEIGHT_GAMMA)
        sw_base = 1.0 + SAMPLE_WEIGHT_MULTIPLIER * scaled_move
        sw_base = np.where(abs_y_raw > HIGH_VOL_THRESHOLD, sw_base * HIGH_VOL_WEIGHT, sw_base)
        sw_base = np.clip(sw_base, 1.0, SAMPLE_WEIGHT_CLIP_MAX).astype(np.float32)
        # [FIX 2b] Augmentation làm tăng số mẫu → sw phải được repeat tương ứng
        # _augment thêm noise vào BẢN COPY, không thêm sample mới → len vẫn như cũ
        # Nhưng để an toàn: align theo len(X_tr_aug)
        n_aug = len(X_tr_aug)
        n_base = len(sw_base)
        if n_aug != n_base:
            # Lặp lại sw_base để khớp (trường hợp augment thêm sample)
            repeat_times = int(np.ceil(n_aug / n_base))
            sw = np.tile(sw_base, repeat_times)[:n_aug]
        else:
            sw = sw_base


    history = model.fit(
        X_tr_aug, Y_smooth,
        sample_weight=sw,
        validation_data=(w["X_va"], w["Y_va_s"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        shuffle=False,              # time-series: KHÔNG shuffle
        verbose=0,
    )

    # Load best checkpoint
    if os.path.exists(ckpt_path):
        try:
            model = tf.keras.models.load_model(ckpt_path, compile=False)
        except Exception as e:
            _warn(f"Không load được checkpoint: {e}")

    ep_run    = len(history.history["loss"])
    best_val  = float(min(history.history.get("val_loss", [float("inf")])))
    _log(f"  [{model_name}] epochs={ep_run} | best_val_loss={best_val:.6f}")

    # Lưu full model + history
    model.save(os.path.join(folder, f"model_{model_name}_{sc}.keras"))
    with open(os.path.join(folder, f"history_{model_name}_{sc}.json"), "w") as f:
        json.dump({
            "loss":     [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
        }, f)

    return {
        "model":     model,
        "best_val":  best_val,
        "ep_run":    ep_run,
    }



# =============================================================================
# STEP 10 — PREDICT & EVALUATE
# =============================================================================

def _inverse_to_price(
    pred_scaled: np.ndarray,
    last_close: np.ndarray,
    scaler_y: StandardScaler,
    return_scale: float = 1.0,
) -> np.ndarray:
    """
    1. Inverse StandardScaler → log-return thực
    2. Close_pred = last_close × exp(cumsum(log_ret))
    """
    bs, h     = pred_scaled.shape
    log_ret   = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(bs, h)
    log_ret   = log_ret * float(return_scale)
    prices    = np.zeros((bs, h), dtype=np.float64)
    for i in range(bs):
        prices[i] = float(last_close[i]) * np.exp(np.cumsum(log_ret[i]))
    return prices


def _select_return_scale(
    pred_scaled: np.ndarray,
    actual: np.ndarray,
    last_close: np.ndarray,
    scaler_y: StandardScaler,
    h: int,
) -> float:
    if not USE_RETURN_CALIBRATION:
        return 1.0

    best_scale = 1.0
    best_score = float("inf")
    naive = _naive_metrics(actual, last_close, h)

    for scale in RETURN_SCALE_CANDIDATES:
        pred = _inverse_to_price(pred_scaled, last_close, scaler_y, return_scale=scale)
        metrics = _metrics(actual, pred)
        metrics["DA"] = _directional_accuracy(actual, pred, last_close)
        anti = _anti_lazy_metrics(actual, pred, last_close, naive, metrics["MAPE"], h)
        score = metrics["MAPE"]
        score += max(0.0, MIN_MOVE_RATIO - anti["MoveRatio"]) * 2.0
        score += max(0.0, anti["CopyRatio"] - PASS_MAX_COPY_RATIO) * 2.0
        score += max(0.0, (MIN_DA_PASS * 100.0) - metrics["DA"]) * 0.03
        if score < best_score:
            best_score = score
            best_scale = float(scale)

    return best_scale


def _actual_price(Y_raw: np.ndarray, last_close: np.ndarray) -> np.ndarray:
    bs, h  = Y_raw.shape
    prices = np.zeros((bs, h), dtype=np.float64)
    for i in range(bs):
        prices[i] = float(last_close[i]) * np.exp(np.cumsum(Y_raw[i]))
    return prices

def _directional_accuracy(actual, pred, last_close):
    a = actual.flatten()
    p = pred.flatten()

    lc = np.repeat(last_close, actual.shape[1])

    true_dir = np.sign(a - lc)
    pred_dir = np.sign(p - lc)

    return float(np.mean(true_dir == pred_dir) * 100)

def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """
    Tính RMSE, MAE, MAPE, R², VolRatio trên giá Close (đã inverse transform).
    DA được tính riêng bởi _directional_accuracy() để đảm bảo so sánh với Last_Close.
    VolRatio = std(diff(pred)) / std(diff(actual)) — đo mức độ biến động dự báo so thực tế.
    """
    a = actual.flatten().astype(np.float64)
    p = pred.flatten().astype(np.float64)
    mask = a != 0
    a_m, p_m = a[mask], p[mask]
    rmse = float(np.sqrt(np.mean((a_m - p_m) ** 2)))
    mae  = float(np.mean(np.abs(a_m - p_m)))
    mape = float(np.mean(np.abs((a_m - p_m) / (np.abs(a_m) + 1e-9))) * 100)
    ss_res = np.sum((a_m - p_m) ** 2)
    ss_tot = np.sum((a_m - np.mean(a_m)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    # VolRatio: dùng diff để đo biến động — nếu model phẳng thì vr → 0
    vr   = float(np.std(np.diff(p_m)) / (np.std(np.diff(a_m)) + 1e-9)) if len(a_m) > 1 else 0.0
    # DA KHÔNG tính ở đây — tính riêng bằng _directional_accuracy() với last_close
    return {"RMSE": round(rmse,6), "MAE": round(mae,6), "MAPE": round(mape,4),
            "R2": round(r2,6), "DA": 0.0, "VolRatio": round(vr,4)}


def _return_metrics(actual: np.ndarray, pred: np.ndarray, last_close: np.ndarray, h: int) -> dict:
    """
    Metrics on cumulative log-return from the window's last close.
    Price-level R2 is useful, but return-level metrics show whether the model
    learned movement instead of simply staying close to Last_Close.
    """
    lc_rep = np.repeat(last_close, h).astype(np.float64)
    a_ret = np.log((actual.flatten().astype(np.float64) + 1e-9) / (lc_rep + 1e-9))
    p_ret = np.log((pred.flatten().astype(np.float64) + 1e-9) / (lc_rep + 1e-9))

    rmse = float(np.sqrt(np.mean((a_ret - p_ret) ** 2)))
    mae = float(np.mean(np.abs(a_ret - p_ret)))
    ss_res = float(np.sum((a_ret - p_ret) ** 2))
    ss_tot = float(np.sum((a_ret - np.mean(a_ret)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    if np.std(a_ret) > 1e-12 and np.std(p_ret) > 1e-12:
        corr = float(np.corrcoef(a_ret, p_ret)[0, 1])
    else:
        corr = 0.0

    return {
        "Return_RMSE": round(rmse, 6),
        "Return_MAE": round(mae, 6),
        "Return_R2": round(r2, 6),
        "ReturnCorr": round(corr, 4),
    }


def _naive_metrics(actual: np.ndarray, last_close: np.ndarray, h: int) -> dict:
    """
    Naive baseline: Pred_Close = Last_Close (repeat h lần).
    Tính RMSE, MAE, MAPE cho baseline này.
    Naive_DA = tỷ lệ các bước thực tế có hướng đúng nếu đoán "không đổi".
    """
    naive = np.tile(last_close.reshape(-1, 1), (1, h))
    a = actual.flatten(); n = naive.flatten(); lc = naive.flatten()
    mask = a != 0
    naive_rmse = round(float(np.sqrt(np.mean((a[mask]-n[mask])**2))), 6)
    naive_mae  = round(float(np.mean(np.abs(a[mask]-n[mask]))), 6)
    naive_mape = round(float(np.mean(np.abs((a[mask]-n[mask])/(np.abs(a[mask])+1e-9)))*100), 4)
    # Naive DA: predict "không đổi" → đúng hướng khi actual == last_close (hiếm)
    true_dir   = np.sign(a - lc)
    naive_dir  = np.zeros_like(lc)  # naive predict 0 return
    naive_da   = round(float(np.mean(true_dir == naive_dir) * 100), 2)
    return {
        "Naive_RMSE": naive_rmse,
        "Naive_MAE":  naive_mae,
        "Naive_MAPE": naive_mape,
        "Naive_DA":   naive_da,
    }


def _anti_lazy_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    last_close: np.ndarray,
    naive: dict,
    model_mape: float,
    h: int,
) -> dict:
    """
    Các metrics chống lazy/copy giá:
    ─────────────────────────────────────────────────────
    Beats_Naive_MAPE : model_MAPE < Naive_MAPE
    NaiveImprove_MAPE: (Naive_MAPE - Model_MAPE) / Naive_MAPE  (> 0 = tốt hơn naive)
    NaiveMAPERatio   : Te_MAPE / Naive_MAPE  (metric cũ, đổi tên — không phải CopyRatio thực)
    CopyRatio        : [FIX] tỷ lệ dự báo có |Pred - Last_Close| / |Last_Close| < COPY_PRICE_REL_THRESHOLD
                       → cao = model đang "copy" giá trước chứ không thực sự dự báo
    MoveRatio        : mean(|pred - last_close|) / mean(|actual - last_close|)
                       → < MIN_MOVE_RATIO = model dự đoán "phẳng"
    LastCloseCorr    : Pearson corr(pred_flat, last_close_rep)
                       → gần 1.0 = model copy giá trước
    ZeroReturnRatio  : % sample có |pred_log_return_approx| < 0.001
                       → phát hiện model luôn predict return ≈ 0

    Lưu ý horizon > 1:
      - last_close được repeat h lần để khớp shape flatten
      - CopyRatio so Pred với Last_Close GỐC (không phải step-by-step) để detect copy-price
      - DA_test (directional accuracy) được tính riêng trong step10 bằng _directional_accuracy()
        cũng dùng Last_Close gốc → nhất quán
    ─────────────────────────────────────────────────────
    """
    a_flat  = actual.flatten().astype(np.float64)
    p_flat  = pred.flatten().astype(np.float64)
    # last_close shape (N,) → repeat h lần → (N*h,) để khớp với p_flat/a_flat
    lc_rep  = np.repeat(last_close, h).astype(np.float64)

    # Beats_Naive_MAPE
    beats_mape = bool(model_mape < naive["Naive_MAPE"])

    # NaiveImprove_MAPE
    naive_improve = round(
        float((naive["Naive_MAPE"] - model_mape) / (naive["Naive_MAPE"] + 1e-9)), 6
    )

    # NaiveMAPERatio: tên mới của "copy_ratio" cũ (Te_MAPE / Naive_MAPE)
    # Giá trị < 1.0 nghĩa là model tốt hơn naive; không phải "copy price"
    naive_mape_ratio = round(float(model_mape) / (float(naive["Naive_MAPE"]) + 1e-9), 4)

    # [FIX] CopyRatio thực sự: % dự báo có |Pred - Last_Close| / |Last_Close| < threshold
    # COPY_PRICE_REL_THRESHOLD = 0.001 (0.1%) — nếu pred gần như bằng last_close → "copy"
    rel_diff  = np.abs(p_flat - lc_rep) / (np.abs(lc_rep) + 1e-9)
    copy_ratio = round(float(np.mean(rel_diff < COPY_PRICE_REL_THRESHOLD)), 4)

    # MoveRatio
    pred_move   = np.mean(np.abs(p_flat - lc_rep))
    actual_move = np.mean(np.abs(a_flat - lc_rep))
    move_ratio  = round(float(pred_move / (actual_move + 1e-9)), 4)

    # LastCloseCorr
    if np.std(p_flat) > 1e-9 and np.std(lc_rep) > 1e-9:
        lcc = float(np.corrcoef(p_flat, lc_rep)[0, 1])
    else:
        lcc = 1.0   # degenerate: pred is constant = last_close
    last_close_corr = round(lcc, 4)

    # ZeroReturnRatio: log-return approx từ pred price so với last_close
    log_ret_pred = np.log((p_flat + 1e-9) / (lc_rep + 1e-9))
    zero_return_ratio = round(float(np.mean(np.abs(log_ret_pred) < 0.001)), 4)

    return {
        "Beats_Naive_MAPE":   beats_mape,
        "NaiveImprove_MAPE":  naive_improve,
        "NaiveMAPERatio":     naive_mape_ratio,   # đổi tên từ "copy_ratio" cũ
        "CopyRatio":          copy_ratio,          # [FIX] định nghĩa đúng
        "MoveRatio":          move_ratio,
        "LastCloseCorr":      last_close_corr,
        "ZeroReturnRatio":    zero_return_ratio,
    }


# =============================================================================
# _build_pred_df — module-level helper (không nằm trong function nào)
# =============================================================================

def _build_pred_df(Y_dates: np.ndarray, actual: np.ndarray, pred: np.ndarray, h: int) -> pd.DataFrame:
    """
    Tạo DataFrame dự báo từ mảng raw.

    Args:
        Y_dates : shape (N, h) hoặc (N,) — datetime/date của target
        actual  : shape (N, h) — giá thực (đã inverse transform)
        pred    : shape (N, h) — giá dự báo
        h       : horizon (1, 3, 7)

    Returns:
        DataFrame với cột Date, Actual, Predicted (trung bình theo ngày nếu trùng).
        Với h > 1 thêm cột Step (bước dự báo).

    Note horizon > 1:
        DA được tính so với Last_Close gốc (Lc) trong _directional_accuracy().
        _build_pred_df chỉ ghi nhận giá trị — không tính direction.
        Group-by Date lấy mean để xử lý trùng ngày khi h > 1.
    """
    rows = []
    if h == 1:
        for d, a, p in zip(Y_dates.flatten(), actual.flatten(), pred.flatten()):
            rows.append({
                "Date":      pd.Timestamp(d).date(),
                "Actual":    float(a),
                "Predicted": float(p),
            })
    else:
        for i in range(len(Y_dates)):
            for j in range(h):
                rows.append({
                    "Date":      pd.Timestamp(Y_dates[i][j]).date(),
                    "Actual":    float(actual[i][j]),
                    "Predicted": float(pred[i][j]),
                    "Step":      j + 1,
                })
    return (
        pd.DataFrame(rows)
        .groupby("Date", as_index=False)
        .mean(numeric_only=True)
        .sort_values("Date")
        .reset_index(drop=True)
        .round(4)
    )


def step10_predict_evaluate(
    model_result: dict,
    w: dict,
    scenario: dict,
    model_name: str,
    ticker: str,
    folder: str,
) -> dict:
    """
    Dự báo train/val/test, inverse transform → giá Close,
    tính metrics 3 tập.
    """
    model    = model_result["model"]
    scaler_y = w["scaler_y"]
    h        = scenario["horizon"]
    sc       = scenario["name"]

    pred_tr_s = _mc_predict(model, w["X_tr"])
    pred_va_s = _mc_predict(model, w["X_va"])
    pred_te_s = _mc_predict(model, w["X_te"])

    act_tr  = _actual_price(w["Y_tr"], w["Lc_tr"])
    act_va  = _actual_price(w["Y_va"], w["Lc_va"])
    act_te  = _actual_price(w["Y_te"], w["Lc_te"])

    return_scale = _select_return_scale(pred_va_s, act_va, w["Lc_va"], scaler_y, h)
    pred_tr = _inverse_to_price(pred_tr_s, w["Lc_tr"], scaler_y, return_scale=return_scale)
    pred_va = _inverse_to_price(pred_va_s, w["Lc_va"], scaler_y, return_scale=return_scale)
    pred_te = _inverse_to_price(pred_te_s, w["Lc_te"], scaler_y, return_scale=return_scale)

    m_tr    = _metrics(act_tr, pred_tr)
    m_va    = _metrics(act_va, pred_va)
    m_te    = _metrics(act_te, pred_te)
    r_tr    = _return_metrics(act_tr, pred_tr, w["Lc_tr"], h)
    r_va    = _return_metrics(act_va, pred_va, w["Lc_va"], h)
    r_te    = _return_metrics(act_te, pred_te, w["Lc_te"], h)
    naive   = _naive_metrics(act_te, w["Lc_te"], h)
    
    m_tr["DA"] = _directional_accuracy(act_tr, pred_tr, w["Lc_tr"])
    m_va["DA"] = _directional_accuracy(act_va, pred_va, w["Lc_va"])
    m_te["DA"] = _directional_accuracy(act_te, pred_te, w["Lc_te"])

    # ── Anti-lazy / anti-copy metrics (chỉ test set) ─────────────────────────
    anti_lazy = _anti_lazy_metrics(
        act_te, pred_te, w["Lc_te"], naive, m_te["MAPE"], h
    )
    
    # Lưu prediction CSV (test)
    df_pred = _build_pred_df(w["Yd_te"], act_te, pred_te, h)
    df_pred.to_csv(os.path.join(folder, f"pred_{model_name}_{sc}.csv"), index=False)

    _log(f"  [{model_name}|{sc}] "
         f"Tr MAPE={m_tr['MAPE']:.2f}% | Va MAPE={m_va['MAPE']:.2f}% | "
         f"Te MAPE={m_te['MAPE']:.2f}% | Naive={naive['Naive_MAPE']:.2f}% | "
         f"MoveRatio={anti_lazy['MoveRatio']:.3f} | LCCorr={anti_lazy['LastCloseCorr']:.3f} | "
         f"ReturnScale={return_scale:.2f}")

    return {
        "ticker": ticker, 
        "model": model_name, 
        "scenario": sc,
        "m_tr": m_tr, 
        "m_va": m_va, 
        "m_te": m_te, 
        "r_tr": r_tr,
        "r_va": r_va,
        "r_te": r_te,
        "naive": naive,
        "anti_lazy": anti_lazy,
        "return_scale": return_scale,
        "pred_te": pred_te, 
        "act_te": act_te,
        "pred_tr": pred_tr, 
        "act_tr": act_tr,
        "Yd_te": w["Yd_te"], 
        "Lc_te": w["Lc_te"],
        "pred_va": pred_va,
        "act_va": act_va,
        "Yd_va": w["Yd_va"],
        "Lc_va": w["Lc_va"],
    }


# =============================================================================
# STEP 11 — OVERFITTING / LAZY / COPY PRICE AUDIT
# =============================================================================

def step11_overfitting_audit(ev: dict) -> dict:
    """
    Build diagnostics for the visual report.
    This function does not assign PASS/FAIL labels; it only returns metrics and
    RiskFlags that help identify overfitting, copy-price, lazy movement, and
    weak directional accuracy in charts.
    """
    m_tr  = ev["m_tr"]; m_va = ev["m_va"]; m_te = ev["m_te"]
    naive = ev["naive"]
    al    = ev.get("anti_lazy", {})

    # ── Core gap metrics ──────────────────────────────────────────────────────
    train_test_r2_gap = m_tr["R2"] - m_te["R2"]
    train_val_r2_gap = m_tr["R2"] - m_va["R2"]
    val_test_r2_gap = m_va["R2"] - m_te["R2"]
    mape_val_test_gap = (m_te["MAPE"] - m_va["MAPE"]) / (abs(m_va["MAPE"]) + 1e-9)

    r_tr = ev.get("r_tr", {})
    r_va = ev.get("r_va", {})
    r_te = ev.get("r_te", {})
    return_train_test_r2_gap = r_tr.get("Return_R2", 0.0) - r_te.get("Return_R2", 0.0)
    return_val_test_r2_gap = r_va.get("Return_R2", 0.0) - r_te.get("Return_R2", 0.0)
    return_corr_test = r_te.get("ReturnCorr", 0.0)

    # ── Naive comparison ─────────────────────────────────────────────────────
    beats_rmse = bool(m_te["RMSE"] < naive["Naive_RMSE"])
    beats_mape = al.get("Beats_Naive_MAPE", m_te["MAPE"] < naive["Naive_MAPE"])

    # [FIX] CopyRatio thực sự (từ _anti_lazy_metrics đã tính đúng)
    copy_ratio = al.get("CopyRatio", 1.0)          # % pred ≈ Last_Close
    naive_mape_ratio = al.get("NaiveMAPERatio", 1.0)  # Te_MAPE / Naive_MAPE (cũ)
    move_ratio = al.get("MoveRatio", 0.0)
    lc_corr    = al.get("LastCloseCorr", 0.0)
    zero_ret   = al.get("ZeroReturnRatio", 0.0)

    # ── LazyRatio: % predictions gần bằng Last_Close với ngưỡng rất chặt (1e-4) ──
    # Khác CopyRatio (ngưỡng 0.1%) — đây là "near-exact copy" (0.01%)
    pred_flat = ev["pred_te"].flatten()
    lc_rep    = np.repeat(ev["Lc_te"], ev["pred_te"].shape[1])
    diff_r    = np.abs(pred_flat - lc_rep) / (np.abs(lc_rep) + 1e-9)
    lazy_cnt  = int((diff_r < 1e-4).sum())
    lazy_ratio = lazy_cnt / max(len(pred_flat), 1)

    # DA_test: đã tính trong step10, giá trị là % (0-100)
    da_test = m_te["DA"]          # % (e.g. 54.2)
    da_frac = da_test / 100.0     # chuyển về tỷ lệ 0-1 để so với MIN_DA_PASS

    # ── Lazy conditions — dùng để điền LazyDetail ────────────────────────────
    risk_flags = []
    if not beats_rmse:
        risk_flags.append("not_beats_rmse")
    if not beats_mape:
        risk_flags.append("not_beats_mape")
    if copy_ratio > PASS_MAX_COPY_RATIO:
        risk_flags.append(f"high_copy_ratio={copy_ratio:.3f}>{PASS_MAX_COPY_RATIO}")
    if move_ratio < MIN_MOVE_RATIO:
        risk_flags.append(f"low_move_ratio={move_ratio:.3f}<{MIN_MOVE_RATIO}")
    if lazy_ratio > MAX_LAZY_RATIO:
        risk_flags.append(f"high_lazy_ratio={lazy_ratio:.3f}>{MAX_LAZY_RATIO}")
    if zero_ret > 0.50:
        risk_flags.append(f"high_zero_return_ratio={zero_ret:.3f}")
    if da_frac < MIN_DA_PASS:
        risk_flags.append(f"low_DA={da_test:.1f}%<{MIN_DA_PASS*100:.0f}%")

    # Signals are stored for visualization only; they are not model grades.
    overfit_signal = bool((
        val_test_r2_gap > OVERFIT_R2_GAP_THRESHOLD
        and mape_val_test_gap > 0.25
    ) or (
        return_val_test_r2_gap > OVERFIT_R2_GAP_THRESHOLD
        and mape_val_test_gap > 0.25
    ))
    copy_signal = bool(copy_ratio > PASS_MAX_COPY_RATIO)
    lazy_signal = bool(
        move_ratio < MIN_MOVE_RATIO
        or lazy_ratio > MAX_LAZY_RATIO
        or zero_ret > 0.50
    )
    direction_signal = bool(da_frac < MIN_DA_PASS)
    naive_signal = bool((not beats_rmse) or (not beats_mape))
    if overfit_signal:
        risk_flags.append("overfit_signal")

    result = {
        # Naive metrics
        "Naive_RMSE":         naive["Naive_RMSE"],
        "Naive_MAPE":         naive["Naive_MAPE"],
        "Beats_Naive":        beats_rmse,
        "Beats_Naive_MAPE":   beats_mape,
        # Anti-lazy metrics
        "NaiveImprove_MAPE":  al.get("NaiveImprove_MAPE", 0.0),
        "NaiveMAPERatio":     naive_mape_ratio,   # Te_MAPE / Naive_MAPE
        "CopyRatio":          copy_ratio,          # [FIX] % pred ≈ Last_Close
        "LazyRatio":          round(lazy_ratio, 4),
        "MoveRatio":          move_ratio,
        "VolRatio":           m_te["VolRatio"],
        "ZeroReturnRatio":    zero_ret,
        "LastCloseCorr":      lc_corr,
        "DA_test":            da_test,
        # Gap metrics
        "R2_gap":             round(val_test_r2_gap, 4),
        "TrainTest_R2_gap":   round(train_test_r2_gap, 4),
        "TrainVal_R2_gap":    round(train_val_r2_gap, 4),
        "ValR2_gap":          round(train_val_r2_gap, 4),
        "ValTest_R2_gap":     round(val_test_r2_gap, 4),
        "MAPE_ValTest_gap":   round(mape_val_test_gap, 4),
        "Return_R2_gap":      round(return_val_test_r2_gap, 4),
        "ReturnTrainTest_R2_gap": round(return_train_test_r2_gap, 4),
        "ReturnValTest_R2_gap":   round(return_val_test_r2_gap, 4),
        "ReturnCorr_test":    round(return_corr_test, 4),
        # Visual-report diagnostics, not pass/fail grading
        "OverfitSignal":      overfit_signal,
        "CopySignal":         copy_signal,
        "LazySignal":         lazy_signal,
        "DirectionSignal":    direction_signal,
        "NaiveSignal":        naive_signal,
        "RiskFlags":          "|".join(risk_flags) if risk_flags else "OK",
        "LazyDetail":         "|".join(risk_flags) if risk_flags else "OK",
    }

    _log(f"  Audit [{ev['model']}|{ev['scenario']}]: "
         f"ValTest_R2_gap={val_test_r2_gap:.4f} TrainTest_R2_gap={train_test_r2_gap:.4f} "
         f"ReturnValTest_R2_gap={return_val_test_r2_gap:.4f} DA={da_test:.1f}% "
         f"MoveRatio={move_ratio:.3f} CopyRatio={copy_ratio:.3f}(real) "
         f"NaiveMAPERatio={naive_mape_ratio:.3f} "
         f"LCCorr={lc_corr:.3f} ZeroRet={zero_ret:.3f} "
         f"Beats(RMSE/MAPE)={beats_rmse}/{beats_mape}")
    if risk_flags:
        _log(f"    RiskFlags: {', '.join(risk_flags)}")
    return result


# =============================================================================
# STEP 12 — ENSEMBLE + SAVE ALL RESULTS
# =============================================================================

def step12_ensemble_and_save(
    ticker: str,
    scenario: dict,
    ev_dict: dict,          # {model_name: eval_result}
    audit_dict: dict,       # {model_name: audit_result}
    folder: str,
    all_rows: list,         # danh sách metrics rows (append vào đây)
) -> None:
    """
    - ensemble is disabled
    - only save per-model metrics
    - Append all metrics rows into all_rows
    """
    sc = scenario["name"]
    h  = scenario["horizon"]

    # ── Per-model rows ───────────────────────────────────────────────────────
    for mn, ev in ev_dict.items():
        aud = audit_dict.get(mn, {})
        al  = ev.get("anti_lazy", {})
        row = {
            "Ticker": ticker, "Scenario": sc, "Model": mn,
            **{f"Tr_{k}": v for k, v in ev["m_tr"].items()},
            **{f"Va_{k}": v for k, v in ev["m_va"].items()},
            **{f"Te_{k}": v for k, v in ev["m_te"].items()},
            **{f"Tr_{k}": v for k, v in ev.get("r_tr", {}).items()},
            **{f"Va_{k}": v for k, v in ev.get("r_va", {}).items()},
            **{f"Te_{k}": v for k, v in ev.get("r_te", {}).items()},
            **ev["naive"],
            **al,    # NaiveImprove_MAPE, MoveRatio, LastCloseCorr, ZeroReturnRatio, Beats_Naive_MAPE
            **aud,   # R2_gap, CopyRatio, Beats_Naive, RiskFlags, diagnostics
        }
        all_rows.append(row)

    # ── Ensemble đã bị tắt — chỉ lưu per-model metrics ─────────────────────



# =============================================================================
# HELPERS
# =============================================================================

def _banner(msg: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")

def _log(msg: str)  -> None: print(f"  {msg}")
def _warn(msg: str) -> None: print(f"  [WARN] {msg}")


def _save_all_metrics(all_rows: list, results_dir: str) -> pd.DataFrame:
    if not all_rows:
        _warn("Không có metrics để lưu!")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    path = os.path.join(results_dir, "merged_metrics_ALL.csv")
    df.to_csv(path, index=False)
    _log(f"Lưu tổng hợp → {path}  ({len(df)} rows)")

    # Lưu riêng từng ticker
    for ticker, grp in df.groupby("Ticker"):
        p = os.path.join(results_dir, ticker, f"metrics_{ticker}.csv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        grp.to_csv(p, index=False)

    # Visual diagnostic summary for reports. This does not grade PASS/FAIL.
    if "Model" in df.columns and "Scenario" in df.columns:
        try:
            summary_rows = []
            for (model, scenario), grp in df.groupby(["Model", "Scenario"]):
                n_total = len(grp)
                row = {
                    "Model":    model,
                    "Scenario": scenario,
                    "N_Total":  n_total,
                    "Avg_MoveRatio":  round(grp["MoveRatio"].mean(), 4) if "MoveRatio" in grp else None,
                    "Avg_CopyRatio":  round(grp["CopyRatio"].mean(), 4) if "CopyRatio" in grp else None,
                    "Avg_DA_test":    round(grp["DA_test"].mean(), 2) if "DA_test" in grp else None,
                    "Avg_NaiveImprove_MAPE": round(grp["NaiveImprove_MAPE"].mean(), 4)
                                             if "NaiveImprove_MAPE" in grp else None,
                    "Pct_BeatNaiveMAPE": round(grp["Beats_Naive_MAPE"].mean() * 100, 1)
                                         if "Beats_Naive_MAPE" in grp else None,
                    "Avg_TrainTest_R2_gap": round(grp["TrainTest_R2_gap"].mean(), 4)
                                             if "TrainTest_R2_gap" in grp else None,
                    "Avg_ValTest_R2_gap": round(grp["ValTest_R2_gap"].mean(), 4)
                                           if "ValTest_R2_gap" in grp else None,
                    "Pct_OverfitSignal": round(grp["OverfitSignal"].mean() * 100, 1)
                                         if "OverfitSignal" in grp else None,
                    "Pct_LazySignal": round(grp["LazySignal"].mean() * 100, 1)
                                      if "LazySignal" in grp else None,
                    "Pct_CopySignal": round(grp["CopySignal"].mean() * 100, 1)
                                      if "CopySignal" in grp else None,
                    "Pct_DirectionSignal": round(grp["DirectionSignal"].mean() * 100, 1)
                                           if "DirectionSignal" in grp else None,
                }
                summary_rows.append(row)

            df_summary = pd.DataFrame(summary_rows).sort_values(
                ["Model", "Scenario"]
            ).reset_index(drop=True)
            sp = os.path.join(results_dir, "visual_diagnostics_summary.csv")
            df_summary.to_csv(sp, index=False)
            _log(f"Lưu summary  → {sp}")
        except Exception as e:
            _warn(f"Không tạo được visual_diagnostics_summary: {e}")

    return df


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ["Ticker","Scenario","Model",
            "Te_MAPE","Te_R2","Te_DA",
            "NaiveImprove_MAPE","MoveRatio","CopyRatio","LastCloseCorr",
            "Beats_Naive","Beats_Naive_MAPE","DA_test",
            "TrainTest_R2_gap","ValTest_R2_gap",
            "OverfitSignal","LazySignal","CopySignal","DirectionSignal",
            "RiskFlags"]
    cols = [c for c in cols if c in df.columns]

    _banner("TOP KET QUA CHAN DOAN (sort by Te_MAPE)")
    shown = df.sort_values("Te_MAPE") if "Te_MAPE" in df.columns else df
    print(shown[cols].head(20).to_string(index=False))

    summary_cols = [
        "MoveRatio", "CopyRatio", "DA_test", "NaiveImprove_MAPE",
        "TrainTest_R2_gap", "ValTest_R2_gap",
        "OverfitSignal", "LazySignal", "CopySignal", "DirectionSignal",
    ]
    existing = [c for c in summary_cols if c in df.columns]
    if existing:
        _banner("TOM TAT CHAN DOAN THEO MODEL / SCENARIO")
        summary = df.groupby(["Model", "Scenario"])[existing].mean(numeric_only=True).round(4)
        print(summary.reset_index().to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main():
    _banner("VN30 STOCK PRICE PREDICTION — 12-STEP PIPELINE")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    runtime_rows = []
    pipeline_start = time.perf_counter()

    # ── STEP 01 ───────────────────────────────────────────────────────────────
    df_raw = step01_load_validate(DATA_FILE)

    # ── STEP 02 ───────────────────────────────────────────────────────────────
    df_clean = step02_handle_missing(df_raw)

    # ── STEP 03 — Feature engineering per ticker ─────────────────────────────
    # FE is still done per-ticker (rolling indicators, no look-ahead).
    # train/val/test splitting is done GLOBALLY afterward.
    feat_dict = step03_feature_engineering(df_clean)

    # ── STEP 04 ───────────────────────────────────────────────────────────────
    clean_dict = step04_clean_features(feat_dict)

    # ── COMBINE all tickers into one DataFrame ────────────────────────────────
    _banner("COMBINE ALL TICKERS → SINGLE DATAFRAME")
    df_combined = pd.concat(list(clean_dict.values()), ignore_index=True)
    df_combined["TradingDate"] = pd.to_datetime(df_combined["TradingDate"])
    df_combined = df_combined.sort_values(["Ticker", "TradingDate"]).reset_index(drop=True)
    _log(f"Combined shape : {df_combined.shape}")
    _log(f"Tickers        : {df_combined['Ticker'].nunique()}")

    # ── STRICT COMMON DATE FILTER (Option A) ─────────────────────────────────
    _banner("STRICT COMMON DATE FILTER (Option A)")
    n_original_dates = df_combined["TradingDate"].nunique()

    if USE_STRICT_COMMON_DATES:
        df_combined, ticker_count_report = keep_common_trading_dates(
            df_combined,
            date_col="TradingDate",
            ticker_col="Ticker",
            expected_tickers=EXPECTED_TICKER_COUNT,
        )

        # Save common-date report
        ticker_count_report["is_common_date"] = (
            ticker_count_report["n_tickers"] == EXPECTED_TICKER_COUNT
        )
        os.makedirs(os.path.dirname(COMMON_DATE_REPORT_PATH), exist_ok=True)
        ticker_count_report.to_csv(COMMON_DATE_REPORT_PATH, index=False,
                                   encoding="utf-8-sig")
        _log(f"Saved common_date_report → {COMMON_DATE_REPORT_PATH}")
    else:
        _log("USE_STRICT_COMMON_DATES=False — skipping common-date filter")

    n_common_dates  = df_combined["TradingDate"].nunique()
    n_removed_dates = n_original_dates - n_common_dates

    # ── GLOBAL DATE SPLIT ─────────────────────────────────────────────────────
    _banner("GLOBAL DATE SPLIT")
    train_end_date, val_end_date, test_start_date = get_global_split_dates(
        df_combined,
        date_col="TradingDate",
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # Apply to full combined dataframe and run validation
    global_train_df, global_val_df, global_test_df = apply_global_date_split(
        df_combined, train_end_date, val_end_date, date_col="TradingDate"
    )

    # ── VALIDATION CHECKS ────────────────────────────────────────────────────
    # NOTE: USE_TEST_MODE only limits which tickers are used for training/testing
    # (speed). Strict common-date filtering always uses EXPECTED_TICKER_COUNT=30
    # so the common trading calendar is derived from the full VN30 universe,
    # not just the test subset.
    all_tickers_combined = sorted(df_combined["Ticker"].unique().tolist())
    expected_for_validation = EXPECTED_TICKER_COUNT if USE_STRICT_COMMON_DATES else len(all_tickers_combined)
    _validate_global_split(
        global_train_df, global_val_df, global_test_df,
        expected_tickers=expected_for_validation,
    )

    # ── SAVE GLOBAL SPLIT SUMMARY ────────────────────────────────────────────
    save_global_split_summary(
        global_train_df, global_val_df, global_test_df,
        output_path=SPLIT_SUMMARY_PATH,
    )

    # Rebuild clean_dict from the globally filtered + combined dataframe
    clean_dict = {
        ticker: grp.reset_index(drop=True)
        for ticker, grp in df_combined.groupby("Ticker")
    }

    if USE_TEST_MODE:
        tickers = [t for t in TEST_TICKERS if t in clean_dict]
        missing = [t for t in TEST_TICKERS if t not in clean_dict]
        if missing:
            _warn(f"[TEST MODE] Không tìm thấy ticker: {missing}")
        _log(f"[TEST MODE] Chỉ chạy thử: {tickers}")
    else:
        tickers = sorted(clean_dict.keys())

    total = len(tickers)
    _log(f"Bắt đầu train: {total} tickers")

    for idx, ticker in enumerate(tickers, 1):
        ticker_start = time.perf_counter()
        _banner(f"[{idx:02d}/{total}] TICKER: {ticker}")
        df_tick = clean_dict[ticker]

        for scenario in SCENARIOS:
            sc   = scenario["name"]
            _log(f"\n  ── {scenario['label']} ──")

            folder = os.path.join(RESULTS_DIR, ticker, sc)
            os.makedirs(folder, exist_ok=True)

            # ── STEP 05 — use global date boundaries ─────────────────────────
            _banner(f"STEP 05 — SPLIT [{ticker}|{sc}]")
            df_tr, df_va, df_te, split_info = step05_split(
                df_tick,
                train_end_date=train_end_date,
                val_end_date=val_end_date,
            )

            # ── STEP 06 ──────────────────────────────────────────────────────
            _banner(f"STEP 06 — SAVE PRE-TRAIN [{ticker}|{sc}]")
            step06_save_pretrain(ticker, scenario, df_tick, df_tr, df_va, df_te, split_info, RESULTS_DIR)

            # ── STEP 07 ── Scaler fit ONLY on train to prevent leakage ───────
            _banner(f"STEP 07 — SCALE [{ticker}|{sc}]")
            (scaler_x,
             tr_sc, va_sc, te_sc,
             tr_dt, va_dt, te_dt,
             raw_tr, raw_va, raw_te) = step07_scale(df_tr, df_va, df_te, folder)

            # ── STEP 08 ──────────────────────────────────────────────────────
            _banner(f"STEP 08 — SLIDING WINDOWS [{ticker}|{sc}]")
            w = step08_sliding_windows(
                tr_sc, va_sc, te_sc,
                raw_tr, raw_va, raw_te,
                tr_dt, va_dt, te_dt,
                scenario, folder,
            )
            if w is None:
                continue

            ev_dict    = {}
            audit_dict = {}

            for model_name in MODEL_NAMES:
                # ── STEP 09 ──────────────────────────────────────────────────
                _banner(f"STEP 09 — TRAIN {model_name} [{ticker}|{sc}]")
                model_result = step09_train(w, model_name, scenario, folder)

                if model_result is None:
                    continue

                # ── STEP 10 ──────────────────────────────────────────────────
                _banner(f"STEP 10 — EVALUATE {model_name} [{ticker}|{sc}]")
                ev = step10_predict_evaluate(model_result, w, scenario, model_name, ticker, folder)
                ev_dict[model_name] = ev

                # ── STEP 11 ──────────────────────────────────────────────────
                _banner(f"STEP 11 — AUDIT {model_name} [{ticker}|{sc}]")
                aud = step11_overfitting_audit(ev)
                audit_dict[model_name] = aud

                del model_result["model"]
                tf.keras.backend.clear_session()

            # ── STEP 12 ──────────────────────────────────────────────────────
            _banner(f"STEP 12 — ENSEMBLE & SAVE [{ticker}|{sc}]")
            step12_ensemble_and_save(ticker, scenario, ev_dict, audit_dict, folder, all_rows)

        ticker_seconds = time.perf_counter() - ticker_start
        runtime_rows.append({
            "Ticker": ticker,
            "Seconds": round(ticker_seconds, 3),
            "Minutes": round(ticker_seconds / 60.0, 3),
        })
        _log(f"Hoan thanh {ticker} | runtime={ticker_seconds / 60.0:.2f} minutes")

    # ── Lưu tổng hợp ─────────────────────────────────────────────────────────
    df_all = _save_all_metrics(all_rows, RESULTS_DIR)
    _print_summary(df_all)
    total_seconds = time.perf_counter() - pipeline_start
    if runtime_rows:
        runtime_rows.append({
            "Ticker": "TOTAL",
            "Seconds": round(total_seconds, 3),
            "Minutes": round(total_seconds / 60.0, 3),
        })
        runtime_path = os.path.join(RESULTS_DIR, "runtime_summary.csv")
        pd.DataFrame(runtime_rows).to_csv(runtime_path, index=False, encoding="utf-8-sig")
        _log(f"Luu runtime summary -> {runtime_path}")

    # ── FINAL EXPLANATION ─────────────────────────────────────────────────────
    _banner("PIPELINE HOÀN TẤT — STRICT COMMON DATES SUMMARY")
    _log(f"Original trading dates          : {n_original_dates:,}")
    _log(f"Common trading dates kept       : {n_common_dates:,}")
    _log(f"Trading dates removed           : {n_removed_dates:,}")
    _log(f"Train date range                : "
         f"{global_train_df['TradingDate'].min().date()} → "
         f"{global_train_df['TradingDate'].max().date()}")
    _log(f"Validation date range           : "
         f"{global_val_df['TradingDate'].min().date()} → "
         f"{global_val_df['TradingDate'].max().date()}")
    _log(f"Test date range                 : "
         f"{global_test_df['TradingDate'].min().date()} → "
         f"{global_test_df['TradingDate'].max().date()}")
    _log(f"All tickers aligned on same split calendar : "
         f"{'YES ' if USE_STRICT_COMMON_DATES else 'NO (legacy mode)'}")
    _log(f"Files modified                  : config.py, main.py")
    _log(f"Reports saved                   :")
    _log(f"  → {COMMON_DATE_REPORT_PATH}")
    _log(f"  → {SPLIT_SUMMARY_PATH}")
    _log(f"Kết quả tại: {os.path.abspath(RESULTS_DIR)}/")



if __name__ == "__main__":
    main()

