# =============================================================================
# config.py
# ToÃ n bá»™ háº±ng sá»‘ táº­p trung táº¡i Ä‘Ã¢y â€” cÃ¡c file khÃ¡c chá»‰ import tá»« Ä‘Ã¢y.
# =============================================================================

import os

# â”€â”€ ÄÆ°á»ng dáº«n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_FILE   = "D:/DACS/3.0/Dataset_VN30_2016_2026.csv"
RESULTS_DIR = "D:/DACS/3.0/results"

# â”€â”€ Dá»¯ liá»‡u â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MIN_TICKER_ROWS = 300   # Bá» qua ticker cÃ³ < 300 rows sau feature engineering

# â”€â”€ Test mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
USE_TEST_MODE = os.getenv("USE_TEST_MODE", "0") == "1" # True = test, False = full run
TEST_TICKERS = [
    t.strip()
    for t in os.getenv("TEST_TICKERS", "VNM,VPB,VRE").split(",")
    if t.strip()
]

# â”€â”€ Chronological split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TRAIN_RATIO = 0.70      # 70% train
VAL_RATIO   = 0.15      # 15% validation
TEST_RATIO  = 0.15      # 15% test (pháº§n cÃ²n láº¡i)

# â”€â”€ Strict Common Dates (Option A) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Chá»‰ giá»¯ láº¡i ngÃ y giao dá»‹ch cÃ³ Ä‘á»§ táº¥t cáº£ tickers â†’ Ä‘áº£m báº£o má»i ticker dÃ¹ng
# cÃ¹ng lá»‹ch giao dá»‹ch vÃ  cÃ¹ng ranh giá»›i train/val/test.
USE_STRICT_COMMON_DATES  = True    # True = báº­t lá»c ngÃ y chung
EXPECTED_TICKER_COUNT    = 30      # Sá»‘ tickers cáº§n cÃ³ máº·t trong má»—i ngÃ y giao dá»‹ch

# ÄÆ°á»ng dáº«n bÃ¡o cÃ¡o â€” dÃ¹ng RESULTS_DIR Ä‘á»ƒ nháº¥t quÃ¡n vá»›i má»i output khÃ¡c
COMMON_DATE_REPORT_PATH  = os.path.join(RESULTS_DIR, "common_date_report.csv")
SPLIT_SUMMARY_PATH       = os.path.join(RESULTS_DIR, "global_split_summary.csv")

# â”€â”€ 3 ká»‹ch báº£n sliding window â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCENARIOS = [
    {"name": "scenario_1", "label": "Scenario 1 (Lookback=20,  Horizon=1)", "lookback": 20,  "horizon": 1},
    {"name": "scenario_2", "label": "Scenario 2 (Lookback=40,  Horizon=3)", "lookback": 40,  "horizon": 3},
    {"name": "scenario_3", "label": "Scenario 3 (Lookback=100, Horizon=7)", "lookback": 100, "horizon": 7},
]

# â”€â”€ Features dÃ¹ng Ä‘á»ƒ train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Bao gá»“m Open, High, Low, Close (4 cá»™t báº¯t buá»™c theo yÃªu cáº§u)
# Loáº¡i bá» DELTA_* (khÃ´ng trong yÃªu cáº§u)
FEATURE_COLS = [
    # 4 cá»™t OHLC báº¯t buá»™c
    "open_close_ratio",
    "high_close_ratio",
    "low_close_ratio",
    # Returns
    "return_1d",
    "log_return_1d",
    # Spread ná»™i ngÃ y
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

# â”€â”€ Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_NAMES   = ["RNN", "LSTM", "GRU" ]

# â”€â”€ Huáº¥n luyá»‡n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# Patience riÃªng cho horizon=3
H3_EARLY_STOPPING_PATIENCE = 14
H3_REDUCE_LR_PATIENCE      = 5
H3_WARMUP_EPOCHS           = 5

# Patience riÃªng cho horizon=7
H7_EARLY_STOPPING_PATIENCE = 14
H7_REDUCE_LR_PATIENCE      = 5

# â”€â”€ Regularization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DROPOUT_RATE = 0.15
L2_LAMBDA    = 1e-4

# ParHybrid dÃ¹ng regularization máº¡nh hÆ¡n
PARHYBRID_L2_OVERRIDE  = 1e-3
PARHYBRID_SPATIAL_DROP = 0.50
PARHYBRID_GRAD_CLIP    = 0.8

# â”€â”€ Chá»‘ng overfitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LABEL_SMOOTH_ALPHA = 0.0    # Giáº£m tá»« 0.10 â†’ 0.02: train signal rÃµ hÆ¡n, loss khÃ´ng bá»‹ má»
AUG_NOISE_STD_X    = 0.004  # Giáº£m tá»« 0.012 â†’ 0.008: khÃ´ng Ã¡t signal khi label smooth Ä‘Ã£ tháº¥p
AUG_NOISE_STD_Y    = 0.0    # Giáº£m tá»« 0.004 â†’ 0.002
AUG_PROB           = 0.25   # Giáº£m tá»« 0.60 â†’ 0.40: augment Ã­t hÆ¡n, model há»c signal rÃµ hÆ¡n
MC_DROPOUT_SAMPLES = 20     # Monte Carlo dropout inference

# Directional loss weights
DIR_LOSS_WEIGHT_H1 = 0.15
DIR_LOSS_WEIGHT_H3 = 0.12
DIR_LOSS_WEIGHT_H7 = 0.10

# Directional loss: pháº¡t khi dá»± Ä‘oÃ¡n sai chiá»u (actual tÄƒng nhÆ°ng pred giáº£m)
# directional_loss = mean( max(0, -sign(y_true) * y_pred) )
USE_DIR_LOSS       = True   # Báº­t directional penalty

# Magnitude constraint
MAG_CONSTRAINT_RATIO  = 0.35
MAG_CONSTRAINT_WEIGHT = 0.08
MIN_PRED_MOVE_STD     = 0.12
ZERO_MOVE_PENALTY_WEIGHT = 0.08
RETURN_CORR_LOSS_WEIGHT  = 0.08

# â”€â”€ NgÆ°á»¡ng Ä‘Ã¡nh giÃ¡ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
OVERFIT_R2_GAP_THRESHOLD = 0.020   # RÂ²_gap > 0.02 â†’ WARN_OVERFIT
DA_PASS_THRESHOLD        = 55.0    # DA% â‰¥ 55% â†’ PASS
DA_WARN_THRESHOLD        = 52.0    # DA% < 52% â†’ WARN_LAZY
VOLRATIO_TARGET_MIN      = 0.85    # VolRatio < 0.85 â†’ WARN
VOLRATIO_TARGET_MAX      = 1.15    # VolRatio > 1.15 â†’ WARN
LAZY_PASS_THRESHOLD      = 0.05    # LazyRatio > 5% â†’ WARN_LAZY
MAPE_MIN_THRESHOLD       = 0.80    # MAPE < 0.8% â†’ suspect lazy
MAPE_MAX_THRESHOLD       = 2.00    # MAPE > 2.0% â†’ too noisy

# Anti-lazy movement thresholds
# MoveRatio = mean(|Pred-LastClose|) / mean(|Actual-LastClose|)
# < MIN_MOVE_RATIO â†’ model predict quÃ¡ pháº³ng (khÃ´ng dÃ¡m "di chuyá»ƒn")
MIN_MOVE_RATIO           = 0.35    # Dá»± bÃ¡o pháº£i cÃ³ Ã­t nháº¥t 35% biÃªn Ä‘á»™ cá»§a thá»±c táº¿

# â”€â”€ [Cáº¢I TIáº¾N 2] Sample weight cho ngÃ y biáº¿n Ä‘á»™ng lá»›n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
USE_SAMPLE_WEIGHTS       = True    # Báº­t/táº¯t sample weighting
SAMPLE_WEIGHT_GAMMA      = 2.0     # MÅ© khuáº¿ch Ä‘áº¡i: weight = (|return| / median) ^ gamma
SAMPLE_WEIGHT_CLIP_MAX   = 5.0    # Clamp weight tá»‘i Ä‘a â€” trÃ¡nh 1 sample chiáº¿m quÃ¡ nhiá»u gradient
SAMPLE_WEIGHT_MULTIPLIER = 4.0    # Há»‡ sá»‘ nhÃ¢n (giáº£m tá»« 10 â†’ 5 Ä‘á»ƒ á»•n Ä‘á»‹nh hÆ¡n)
# NgÆ°á»¡ng phÃ¢n loáº¡i "ngÃ y biáº¿n Ä‘á»™ng lá»›n": |log_return| > HIGH_VOL_THRESHOLD
# NgÃ y thÆ°á»ng â†’ weight=1.0, ngÃ y biáº¿n Ä‘á»™ng lá»›n â†’ weight=HIGH_VOL_WEIGHT
HIGH_VOL_THRESHOLD       = 0.015  # 1.5% log-return
HIGH_VOL_WEIGHT          = 5.0    # Giáº£m tá»« 10 â†’ 5: gradient á»•n Ä‘á»‹nh hÆ¡n, khÃ´ng bá»‹ spike

# â”€â”€ [Cáº¢I TIáº¾N 3] Huber loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
USE_HUBER_LOSS           = True    # True = Huber, False = MSE
HUBER_DELTA              = 1.0     # delta=1.0: gáº§n MAE khi sai sá»‘ lá»›n, MSE khi nhá»

# Diagnostic thresholds used for charts and RiskFlags. They are not PASS/FAIL grades.
PASS_REQUIRE_BEATS_NAIVE = True
PASS_MIN_DA              = 52.0
PASS_REQUIRE_MAPE_BEAT   = True
PASS_MAX_COPY_RATIO      = 0.60

# â”€â”€ [FIX] NgÆ°á»¡ng CopyRatio thá»±c sá»± â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CopyRatio = tá»· lá»‡ dá»± bÃ¡o cÃ³ |Pred_Close - Last_Close| / |Last_Close| < threshold
# KhÃ¡c vá»›i NaiveMAPERatio = Te_MAPE / Naive_MAPE (metric cÅ©, Ä‘á»•i tÃªn)
COPY_PRICE_REL_THRESHOLD = 0.001   # 0.1% â€” náº¿u pred chÃªnh < 0.1% so Last_Close â†’ "copy"

# â”€â”€ [FIX] NgÆ°á»¡ng WARN_LAZY vÃ  MIN_DA_PASS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MAX_LAZY_RATIO           = 0.60    # LazyRatio > 60% â†’ WARN_LAZY
MIN_DA_PASS              = 0.52    # DA_test < 52% â†’ WARN_LOW_DA (tá»· lá»‡, khÃ´ng pháº£i %)

# Validation-time return calibration. This post-processes predicted log-returns
# with one scalar selected on validation data to reduce flat/copy forecasts.
USE_RETURN_CALIBRATION = True
RETURN_SCALE_CANDIDATES = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

# â”€â”€ Seed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42

