# Du doan gia chung khoan VN30

Repo nay chi chua source code cho do an du bao gia dong cua 30 ma VN30 bang RNN, LSTM va GRU.

Data, model da train, results, chart va file PDF bao cao khong duoc dua len GitHub. Cac file nay duoc ignore trong `.gitignore` de repo gon va dung yeu cau "code only".

## File chinh

- `main.py`: pipeline huan luyen va danh gia mo hinh.
- `config.py`: cau hinh feature, scenario, model va hyperparameter.
- `rebuild_metrics.py`: rebuild metrics tu artifact da huan luyen.
- `visualize_model_report.py`: ve bieu do tu `results/merged_metrics_ALL.csv`.
- `CRAWVN30.py`: script thu thap du lieu ban dau bang `vnstock`.
- `requirements.txt`: danh sach thu vien Python can cai.

## Cai dat

```bash
pip install -r requirements.txt
```

## Chay pipeline

Dat file du lieu `Dataset_VN30_2016_2026.csv` vao thu muc goc project, sau do chay:

```bash
python main.py
```

## Rebuild metrics

Khi da co artifact trong thu muc `results/`, co the tinh lai metrics bang:

```bash
python rebuild_metrics.py --results-dir results --write-predictions
```

## Ve bieu do

```bash
python visualize_model_report.py --results-dir results --max-detail-plots 0
```

## Ghi chu

Thu muc `results/`, file data `.csv`, model `.keras`, scaler `.pkl`, anh `.png` va PDF bao cao deu duoc ignore. Neu can tai lap day du ket qua, can co data va chay lai pipeline tren may local.
