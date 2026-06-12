# =============================================================================
# config.py
# Toàn bộ hằng số tập trung tại đây — các file khác chỉ import từ đây.
# =============================================================================

import os

# ── Đường dẫn ────────────────────────────────────────────────────────────────
DATA_FILE   = "D:/DACS/3.0/Dataset_VN30_2016_2026.csv"
RESULTS_DIR = "D:/DACS/3.0/results"

# ── Dữ liệu ──────────────────────────────────────────────────────────────────
MIN_TICKER_ROWS = 300   # Bỏ qua ticker có < 300 rows sau feature engineering

# ── Test mode ────────────────────────────────────────────────────────────────
USE_TEST_MODE = os.getenv("USE_TEST_MODE", "0") == "1" # True = test, False = full run
TEST_TICKERS = [
    t.strip()
    for t in os.getenv("TEST_TICKERS", "VNM,VPB,VRE").split(",")
    if t.strip()
]

# ── Chronological split ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.70      # 70% train
VAL_RATIO   = 0.15      # 15% validation
TEST_RATIO  = 0.15      # 15% test (phần còn lại)

# ── Strict Common Dates (Option A) ────────────────────────────────────────────
# Chỉ giữ lại ngày giao dịch có đủ tất cả tickers → đảm bảo mọi ticker dùng
# cùng lịch giao dịch và cùng ranh giới train/val/test.
USE_STRICT_COMMON_DATES  = True    # True = bật lọc ngày chung
EXPECTED_TICKER_COUNT    = 30      # Số tickers cần có mặt trong mỗi ngày giao dịch

# Đường dẫn báo cáo — dùng RESULTS_DIR để nhất quán với mọi output khác
COMMON_DATE_REPORT_PATH  = os.path.join(RESULTS_DIR, "common_date_report.csv")
SPLIT_SUMMARY_PATH       = os.path.join(RESULTS_DIR, "global_split_summary.csv")

# ── 3 kịch bản sliding window ────────────────────────────────────────────────
SCENARIOS = [
    {"name": "scenario_1", "label": "Scenario 1 (Lookback=20,  Horizon=1)", "lookback": 20,  "horizon": 1},
    {"name": "scenario_2", "label": "Scenario 2 (Lookback=40,  Horizon=3)", "lookback": 40,  "horizon": 3},
    {"name": "scenario_3", "label": "Scenario 3 (Lookback=100, Horizon=7)", "lookback": 100, "horizon": 7},
]

# ── Features dùng để train ────────────────────────────────────────────────────
# Bao gồm Open, High, Low, Close (4 cột bắt buộc theo yêu cầu)
# Loại bỏ DELTA_* (không trong yêu cầu)
FEATURE_COLS = [
    # 4 cột OHLC bắt buộc
    "open_close_ratio",
    "high_close_ratio",
    "low_close_ratio",
    # Returns
    "return_1d",
    "log_return_1d",
    # Spread nội ngày
    "hl_range_ratio",
    "oc_change_ratio",
    # Volume
    "Volume_Change",
    # Moving Averages
    "ma5_close_ratio",
    "ma10_close_ratio",
    "ma20_close_ratio",
    # EMA
    "EMA12",
    "EMA26",
    # Momentum oscillators
    "RSI14",
    "MACD",
    "MACD_Signal",
    # Bollinger Bands
    "bb_upper_close_ratio",
    "bb_lower_close_ratio",
    # Volatility
    "ATR14",
    "atr14_close_ratio",
    # Gap
    "Gap_Flag",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volatility_10d",
    "volume_ma_ratio",
    "price_ma5_gap",
    "price_ma20_gap",
    # Breakout features
    "rolling_max20_close_ratio",
    "rolling_min20_close_ratio",
    "breakout_up",
    "breakout_down",
    "volume_spike",
]

N_FEATURES = len(FEATURE_COLS)

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_NAMES   = ["RNN", "LSTM", "GRU" ]

# ── Huấn luyện ───────────────────────────────────────────────────────────────
EPOCHS     = 80
BATCH_SIZE = 64
INITIAL_LR = 5e-4

# EarlyStopping
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MONITOR  = "val_loss"

# ReduceLROnPlateau
REDUCE_LR_MONITOR  = "val_loss"
REDUCE_LR_FACTOR   = 0.5
REDUCE_LR_PATIENCE = 5
REDUCE_LR_MIN_LR   = 1e-7

# Patience riêng cho horizon=3
H3_EARLY_STOPPING_PATIENCE = 14
H3_REDUCE_LR_PATIENCE      = 5
H3_WARMUP_EPOCHS           = 5

# Patience riêng cho horizon=7
H7_EARLY_STOPPING_PATIENCE = 14
H7_REDUCE_LR_PATIENCE      = 5

# ── Regularization ───────────────────────────────────────────────────────────
DROPOUT_RATE = 0.15
L2_LAMBDA    = 1e-4

# ParHybrid dùng regularization mạnh hơn
PARHYBRID_L2_OVERRIDE  = 1e-3
PARHYBRID_SPATIAL_DROP = 0.50
PARHYBRID_GRAD_CLIP    = 0.8

# ── Chống overfitting ────────────────────────────────────────────────────────
LABEL_SMOOTH_ALPHA = 0.0    # Giảm từ 0.10 → 0.02: train signal rõ hơn, loss không bị mờ
AUG_NOISE_STD_X    = 0.004  # Giảm từ 0.012 → 0.008: không át signal khi label smooth đã thấp
AUG_NOISE_STD_Y    = 0.0    # Giảm từ 0.004 → 0.002
AUG_PROB           = 0.25   # Giảm từ 0.60 → 0.40: augment ít hơn, model học signal rõ hơn
MC_DROPOUT_SAMPLES = 20     # Monte Carlo dropout inference

# Directional loss weights
DIR_LOSS_WEIGHT_H1 = 0.15
DIR_LOSS_WEIGHT_H3 = 0.12
DIR_LOSS_WEIGHT_H7 = 0.10

# Directional loss: phạt khi dự đoán sai chiều (actual tăng nhưng pred giảm)
# directional_loss = mean( max(0, -sign(y_true) * y_pred) )
USE_DIR_LOSS       = True   # Bật directional penalty

# Magnitude constraint
MAG_CONSTRAINT_RATIO  = 0.35
MAG_CONSTRAINT_WEIGHT = 0.08
MIN_PRED_MOVE_STD     = 0.12
ZERO_MOVE_PENALTY_WEIGHT = 0.08
RETURN_CORR_LOSS_WEIGHT  = 0.08

# ── Ngưỡng đánh giá ──────────────────────────────────────────────────────────
OVERFIT_R2_GAP_THRESHOLD = 0.020   # R²_gap > 0.02 → WARN_OVERFIT
DA_PASS_THRESHOLD        = 55.0    # DA% ≥ 55% → PASS
DA_WARN_THRESHOLD        = 52.0    # DA% < 52% → WARN_LAZY
VOLRATIO_TARGET_MIN      = 0.85    # VolRatio < 0.85 → WARN
VOLRATIO_TARGET_MAX      = 1.15    # VolRatio > 1.15 → WARN
LAZY_PASS_THRESHOLD      = 0.05    # LazyRatio > 5% → WARN_LAZY
MAPE_MIN_THRESHOLD       = 0.80    # MAPE < 0.8% → suspect lazy
MAPE_MAX_THRESHOLD       = 2.00    # MAPE > 2.0% → too noisy

# Anti-lazy movement thresholds
# MoveRatio = mean(|Pred-LastClose|) / mean(|Actual-LastClose|)
# < MIN_MOVE_RATIO → model predict quá phẳng (không dám "di chuyển")
MIN_MOVE_RATIO           = 0.35    # Dự báo phải có ít nhất 35% biên độ của thực tế

# ── [CẢI TIẾN 2] Sample weight cho ngày biến động lớn ────────────────────────
USE_SAMPLE_WEIGHTS       = True    # Bật/tắt sample weighting
SAMPLE_WEIGHT_GAMMA      = 2.0     # Mũ khuếch đại: weight = (|return| / median) ^ gamma
SAMPLE_WEIGHT_CLIP_MAX   = 5.0    # Clamp weight tối đa — tránh 1 sample chiếm quá nhiều gradient
SAMPLE_WEIGHT_MULTIPLIER = 4.0    # Hệ số nhân (giảm từ 10 → 5 để ổn định hơn)
# Ngưỡng phân loại "ngày biến động lớn": |log_return| > HIGH_VOL_THRESHOLD
# Ngày thường → weight=1.0, ngày biến động lớn → weight=HIGH_VOL_WEIGHT
HIGH_VOL_THRESHOLD       = 0.015  # 1.5% log-return
HIGH_VOL_WEIGHT          = 5.0    # Giảm từ 10 → 5: gradient ổn định hơn, không bị spike

# ── [CẢI TIẾN 3] Huber loss ───────────────────────────────────────────────────
USE_HUBER_LOSS           = True    # True = Huber, False = MSE
HUBER_DELTA              = 1.0     # delta=1.0: gần MAE khi sai số lớn, MSE khi nhỏ

# Diagnostic thresholds used for charts and RiskFlags. They are not PASS/FAIL grades.
PASS_REQUIRE_BEATS_NAIVE = True
PASS_MIN_DA              = 52.0
PASS_REQUIRE_MAPE_BEAT   = True
PASS_MAX_COPY_RATIO      = 0.60

# ── [FIX] Ngưỡng CopyRatio thực sự ───────────────────────────────────────────
# CopyRatio = tỷ lệ dự báo có |Pred_Close - Last_Close| / |Last_Close| < threshold
# Khác với NaiveMAPERatio = Te_MAPE / Naive_MAPE (metric cũ, đổi tên)
COPY_PRICE_REL_THRESHOLD = 0.001   # 0.1% — nếu pred chênh < 0.1% so Last_Close → "copy"

# ── [FIX] Ngưỡng WARN_LAZY và MIN_DA_PASS ────────────────────────────────────
MAX_LAZY_RATIO           = 0.60    # LazyRatio > 60% → WARN_LAZY
MIN_DA_PASS              = 0.52    # DA_test < 52% → WARN_LOW_DA (tỷ lệ, không phải %)

# Validation-time return calibration. This post-processes predicted log-returns
# with one scalar selected on validation data to reduce flat/copy forecasts.
USE_RETURN_CALIBRATION = True
RETURN_SCALE_CANDIDATES = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42

