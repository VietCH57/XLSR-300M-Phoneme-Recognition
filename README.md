# Phoneme Recognition: XLS-R-300M + CTC

Nhận dạng chuỗi phoneme (ARPAbet + phoneme accent `*`, kiểu L2-ARCTIC) từ audio.

```
waveform 16 kHz
  → XLS-R-300M  [CNN feature encoder (đóng băng) → feature projection → 24 Transformer layers]
  → dropout → Linear(1024 → 79) → log_softmax
  → CTC loss (train) / CTC greedy decoding (infer) → phoneme sequence
```

Backbone: [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m).
Vocab: `configs/vocab.json` gồm 79 nhãn, `[PAD]` (id 78) là CTC blank, `[UNK]` = 77.
`aa` và `aa*` là hai lớp khác nhau.

## Cấu trúc

```
configs/
  default.yaml          hyperparameters mặc định (1x RTX 4090 24 GB)
  vocab.json
src/phonrec/
  config.py             dataclass Config: defaults → YAML → CLI
  data/
    vocab.py            PhonemeTokenizer
    manifest.py         đọc CSV, làm sạch, path_map, lọc domain
    audio.py            đọc wav/mp3/NIST, resample 16 kHz, cache thời lượng
    dataset.py          PhonemeDataset + build_dataset (lọc độ dài, CTC feasibility)
    sampler.py          dynamic batching theo tổng giây audio + oversampling domain
    collate.py          padding
  models/xlsr_ctc.py    XLSRPhonemeCTC
  decoding.py           CTC greedy decoding
  metrics/per.py        PER (S/D/I) theo domain
  engine.py             vòng lặp eval/infer dùng chung
  utils/                logging, seed, scheduler, checkpoint
scripts/
  train.py
  infer.py
  prepare_audio.py      (tùy chọn) chuyển trước audio sang FLAC 16 kHz
  data_stats.py         thống kê dữ liệu theo domain
tests/                  pytest
```

## Cài đặt

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # build CUDA phù hợp driver
pip install -r requirements.txt
pytest -q          # 14 test, chạy CPU, không cần tải model
```

Lần chạy đầu sẽ tải XLS-R-300M từ Hugging Face (~1.2 GB).

## Dữ liệu

Các CSV có 3 cột `audio,text,domain`, nằm tại `/home/mdd/lamnh/MDD-2026/data/stage2_combined/server/`:

| domain | nguồn |
|---|---|
| `native_en` | TIMIT, CommonVoice EN, FLEURS en_us |
| `native_vi` | CommonVoice VI, FLEURS vi_vn, balanced_vi |
| `l2a` | L2-ARCTIC |
| `el_easy` | eLearningMDD (easy, chunked) |

- **Train / dev / test:** `fullnative_el2a_{train,dev,test}_ctc.csv`.
- **Câu thiếu transcript** bị bỏ và có cảnh báo trong log: 14 câu ở train, 1 câu ở test.
- **`[UNK]` trong transcript:** khoảng 163 câu L2-ARCTIC ở train có `[UNK]` ngay trong transcript gốc. Mặc định `[UNK]` được giữ làm một lớp; đổi bằng `--train_unk_policy keep|drop_token|drop_utt`. Dev và test luôn giữ nguyên.
- **CSV không có cột `domain`:** domain được suy từ đường dẫn (`l2arctic` → `l2a`, `eLearningMDD` → `el_easy`, …).

Khuyến nghị chạy một lần trước khi train:

```bash
D=/home/mdd/lamnh/MDD-2026/data/stage2_combined/server
python scripts/data_stats.py --csv $D/fullnative_el2a_{train,dev,test}_ctc.csv      # kiểm tra file thiếu, phân bố thời lượng
python scripts/prepare_audio.py --csv $D/fullnative_el2a_{train,dev,test}_ctc.csv --num_workers 8
```

`prepare_audio.py` giải mã (mp3, NIST…) và resample sẵn sang `cache/audio16k/*.flac`. Train và infer tự dùng bản cache nếu có, giúp CPU (8 vCPU) không phải giải mã mp3 trong lúc train. Nếu bỏ bước này, code vẫn đọc trực tiếp file gốc.

## Train

```bash
python scripts/train.py --config configs/default.yaml --output_dir runs/exp1
```

Mọi khóa trong `configs/default.yaml` đều ghi đè được bằng CLI cùng tên (`python scripts/train.py --help`):

```bash
# chỉ dùng một số tập con (áp dụng cho cả train và dev)
python scripts/train.py --config configs/default.yaml --output_dir runs/en_l2 \
    --domains native_en l2a el_easy

# oversample domain ít dữ liệu, đổi LR / scheduler
python scripts/train.py --config configs/default.yaml --output_dir runs/exp2 \
    --domain_weights l2a=2 el_easy=4 --lr 3e-5 --scheduler tri_stage --max_steps 30000

# chỉ train head trong 2000 step đầu
python scripts/train.py --config configs/default.yaml --freeze_transformer_steps 2000

# tiếp tục từ checkpoint cuối
python scripts/train.py --config runs/exp1/config.yaml --resume_from runs/exp1/last
```

**Tham số chính:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--domains` | tất cả | lọc tập con cho train và dev |
| `--max_steps` | 20000 | số bước optimizer |
| `--max_batch_seconds` | 200 | tổng giây audio (đã padding) mỗi batch |
| `--grad_accum_steps` | 1 | |
| `--lr` / `--head_lr` | 5e-5 / 5e-4 | LR cho backbone / head |
| `--scheduler` | linear | `linear`, `cosine`, `tri_stage`, `constant`; dùng kèm `--warmup_steps` |
| `--max_audio_seconds` | 30 | câu train dài hơn bị bỏ (dev/test không lọc) |
| `--mask_time_prob`, `--layerdrop`, `--*_dropout` | | regularization |
| `--precision` | bf16 | `bf16`, `fp16`, `fp32` |
| `--gradient_checkpointing` | true | |
| `--eval_every` | 1000 | |
| `--select_metric` | `per/ALL` | ví dụ `per/l2a` hoặc `loss` |
| `--save_top_k` | 3 | |
| `--early_stopping_patience` | 0 (tắt) | |
| `--path_map OLD=NEW` | | đổi prefix đường dẫn audio |

**Output trong `runs/exp1/`:**

```
train.log            toàn bộ log, gồm bảng PER theo domain ở mỗi lần eval
config.yaml          config đã dùng
best/                checkpoint tốt nhất theo select_metric → dùng để infer
checkpoints/step_*/  top-k checkpoint
last/                checkpoint mới nhất + trạng thái optimizer (để resume)
```

**Bảng eval mỗi `eval_every` step:**

```
domain         #utt   #phones   PER(%)       S       D       I
--------------------------------------------------------------
native_en      1808     ...
native_vi      1900     ...
l2a             212     ...
el_easy          71     ...
--------------------------------------------------------------
ALL            3991     ...
```

PER = (S + D + I) / N, cộng dồn trên mọi câu (micro). Cách tính này dùng cho cả từng domain và dòng ALL.

## Infer

```bash
python scripts/infer.py --ckpt runs/exp1/best \
    --csv /home/mdd/lamnh/MDD-2026/data/stage2_combined/server/fullnative_el2a_test_ctc.csv
```

- **Nhiều CSV một lần:** `--csv` nhận nhiều file, ví dụ `--csv a.csv b.csv`.
- **Không lọc domain:** mọi câu đọc được đều được decode.
- **Output:** mặc định ghi vào `<ckpt>/infer/<tên_csv>/`, đổi bằng `--output_dir`:
  - `predictions.csv` gồm `row, audio, domain, ref, hyp, per, sub, del, ins, ref_phones`;
  - `report.json` chứa PER theo domain và ALL.
- **Log:** `infer.log` (cùng bảng PER theo domain như khi eval).
- **CSV không có cột `text`:** chỉ decode, không tính PER.

## Ghi chú

- **Gợi ý cho 4090 24 GB:**
  - Nếu gặp OOM, giảm `--max_batch_seconds` (ví dụ 120) và tăng `--grad_accum_steps`. Batch bị OOM sẽ bị bỏ qua và ghi vào log, không làm dừng train.
  - Nếu GPU chờ dữ liệu, chạy `prepare_audio.py` hoặc tăng `--num_workers`, tối đa 8.
- **Resume:** tiếp tục ở đầu epoch kế tiếp, giữ nguyên số step, optimizer, scheduler và lịch sử best/top-k.
- **Nhãn không có trong train:** `m*` không xuất hiện trong train nên mô hình sẽ không dự đoán ra nó (test chỉ có 1 lần).
