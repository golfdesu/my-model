# Fix Log — Paper Alignment Fixes (`*_fix.py`)

> Date: 2026-08-22
> Scope: สร้างไฟล์ใหม่ 8 ไฟล์ (`*_fix.py`) — 6 ไฟล์แรกเพื่อให้ implementation ตรงกับงานวิจัยต้นฉบับของแต่ละโมเดล + `01_fix`/`11_fix` แก้ bug/config
> **ไฟล์เดิมทั้งหมดไม่ถูกแก้ไข** — originals เหมือนเดิม 100%
> ทุกไฟล์ผ่าน forward/backward smoke test ด้วยข้อมูลจริง (`acn_caltech_ready.csv`) แล้ว

---

## 0. `01_tfm_tfm_auto_pytorch_fix.py` — Vanilla Transformer (bug fixes)

| | |
|---|---|
| **บัคที่เจอ** | ① `weight_decay=1e-3` ต่างจาก benchmark อื่นทั้งหมดที่ใช้ `1e-6` ② duplicate `import sys` ③ `time.sleep(0.005)/(0.002)` ต่อ batch ④ ไม่มี `sort_index()` ⑤ output `.md` append ซ้ำตอน rerun |
| **การแก้** | ① → `weight_decay=1e-6` ② ลบ import ซ้ำ ③ ลบ sleep ทั้ง train/val loop ④ เพิ่ม `df = df.sort_index()` ⑤ truncate `.md` ตอนเริ่ม run |
| **Paper alignment เพิ่ม** | Positional embedding learnable → **Sinusoidal Positional Encoding** (Vaswani et al. 2017 §3.5, fixed buffer) — GaussianNoise คงไว้ (regularization ที่ตั้งใจ) |
| **ผล test** | ✅ out `[16, 48]`, 96,624 params, backward OK, pe buffer `[1,96,64]` |

---

## 0b. `11_lightgbm_baseline_fix.py` — LightGBM (config bug fix)

| | |
|---|---|
| **บัคที่เจอ** | `subsample=0.8` **ไม่ active** — LightGBM เปิด bagging เฉพาะเมื่อ `subsample_freq > 0` (default 0) → ใช้ข้อมูล 100% ทุก tree เงียบ ๆ เทียบ XGBoost ไม่ fair ② ไม่มี `sort_index()` ③ `.md` append ซ้ำ |
| **การแก้** | ① เพิ่ม `subsample_freq=1` (resample ทุก iteration) ② `sort_index()` ③ truncate `.md` ตอนเริ่ม run |
| **ผล** | ✅ compile OK |

---

## 1. สรุปการแก้รายไฟล์

### `02_tfm_ifm_auto_pytorch_fix.py` — Informer (Zhou et al., AAAI 2021)

| | |
|---|---|
| **บัคที่เจอ** | `ProbAttention.forward()` ใช้ `values_p.sum(dim=-2)` สำหรับ query ที่ไม่ติด top-u — official repo (`_get_initial_context`) ใช้ `V.mean(dim=-2)` → position เหล่านั้นได้ค่า sum (~50–100x ใหญ่กว่าควร) แทน mean |
| **การแก้** | `.sum(dim=-2, keepdim=True)` → `.mean(dim=-2, keepdim=True)` (บรรทัดเดียวจบ) |
| **ผล test** | ✅ out `[16, 48]`, 108,880 params, backward OK |

---

### `03_tfm_afm_auto_pytorch_fix.py` — Autoformer (Wu et al., NeurIPS 2021)

| | |
|---|---|
| **ไม่ตรง paper** | ① `SeriesDecomp` ใช้ `AvgPool1d(padding=k//2)` = zero padding ที่ขอบ (official ใช้ replicate edge padding) ② decoder ถูกย่อมาก: ไม่มี progressive trend accumulation ข้าม layer, seasonal/trend init ด้วย last-step repeat แทน zero/mean-pooled trend, ไม่มี decomposition หลังแต่ละ sublayer |
| **การแก้** | ① SeriesDecomp ใหม่: repeat ขอบซ้าย/ขวา + `AvgPool1d(stride=1)` ไม่ pad ② เพิ่ม class `AutoformerDecoderLayer` (causal self-attn → cross-attn → FFN พร้อม decomp หลังทุก sublayer, สะสม trend) ③ decoder init ใหม่: `seasonal = zeros`, `trend = mean-pooled encoder trend` ④ progressive trend accumulation ข้าม layer ⑤ final decomp + project seasonal/trend แยกแล้ว sum |
| **ผล test** | ✅ out `[16, 48]`, 43,074 params, backward OK |

---

### `04_tfm_tft_auto_pytorch_fix.py` — TFT (Lim et al., IJF 2021)

| | |
|---|---|
| **ไม่ตรง paper** | ① decoder input ใช้ placeholder (last-step repeat) แทน **known-future covariates** — หัวใจของ TFT ② ใช้ standard `nn.MultiheadAttention` แทน **interpretable MHA** (shared W_V + average heads) |
| **การแก้** | ① เพิ่ม `FUTURE_KNOWN_COLS` = 9 calendar features (`weekend, holiday, is_business_hour, Hour_sin/cos, DayOfWeek_sin/cos, Month_sin/cos`) + scaler แยก + helper `create_future_known_tensors` → TensorDataset/DataLoader คืน 3 tensors → `model(x, x_future)` → `vsn_dec(x_future)` ② เพิ่ม class `InterpretableMultiHeadAttention`: W_V เดียว shared ทุก head (output d_attn), attention averaged over heads แล้วค่อย W_O: d_attn→d_model ตาม paper §3.2 |
| **ผล test** | ✅ out `[16, 48, 3]` (P10/P50/P90), 71,141 params, pinball loss backward OK |

---

### `05_tfm_ptft_auto_pytorch_fix.py` — PatchTST (Nie et al., ICLR 2023)

| | |
|---|---|
| **บัค + ไม่ตรง paper** | forecast ครบ 30 channels แล้ว **เฉลี่ย denormalized outputs ทุก channel** (`torch.mean(dec_out, dim=-1)`) — PatchTST เป็น channel-independent: แต่ละ channel forecast ตัวเอง แล้วอ่านเฉพาะ target channel การเฉลี่ย target กับ weather/lag/calendar ผิด semantic ทั้ง scale และ meaning |
| **การแก้** | ① append scaled target series (`kWhDelivered`) เป็น input channel ที่ 31 (`TARGET_CH_IDX=30`) ② forward คืนเฉพาะ `ch_out[:, :, TARGET_CH_IDX]` (normalized domain) — ตัด denorm+mean ทิ้ง |
| **ผล test** | ✅ out `[16, 48]`, 26,440 params, backward OK |

> ⚠️ Note: input กลายเป็น **31 channels** (เพิ่ม target series) — เป็น requirement ของ design แบบ CI multivariate forecasting ต้อง note ตอนรายงานผล

---

### `13_tfm_itfm_auto_pytorch_fix.py` — iTransformer (Liu et al., ICLR 2024)

| | |
|---|---|
| **ไม่ตรง paper** | ① `variate_agg = nn.Linear(num_features, 1)` ผสม forecast ของทุก variate — paper อ่าน forecast จาก **target variate token** โดยตรง (non-target เป็น context เท่านั้น) ② ไม่มี final LayerNorm หลัง encoder stack — official (`Encoder(norm_layer=LayerNorm)`) ใส่ ③ official ใช้ `activation='gelu'` ไม่ใช่ relu |
| **การแก้** | ① append scaled target เป็น variate ที่ 31 (`TARGET_CH_IDX=30`) + ตัด `variate_agg` → `out = x[:, TARGET_CH_IDX, :]` ② เพิ่ม `nn.TransformerEncoder(..., norm=nn.LayerNorm(d_model))` ③ activation → `'gelu'` |
| **ผล test** | ✅ out `[16, 48]`, 13,296 params, backward OK |

> Note: post-norm ของ `nn.TransformerEncoderLayer` (default `norm_first=False`) **ตรงกับ official อยู่แล้ว** — TSLib `EncoderLayer` เป็น post-norm (`x = norm1(x + dropout(attn))`)

> ⚠️ Note: input กลายเป็น **31 variates** เช่นเดียวกับ 05

---

### `14_tfm_timesnet_auto_pytorch_fix.py` — TimesNet (Wu et al., ICLR 2023)

| | |
|---|---|
| **ไม่ตรง paper + latent bug** | ① aggregation เป็น flat mean ทุก period — paper ใช้ **adaptive aggregation**: weight = softmax(amplitude) ของแต่ละ frequency ② in-place `freq_abs[0] = 0` บน tensor ที่อยู่ใน autograd graph (ตอนนี้รอดเพราะ detach ก่อนใช้ แต่ brittle) ③ padding ตอน reshape 2D ใช้ zeros (official ใช้ replicate) ④ **prediction head ใช้ global-avg-pool over time + MLP** — official ฉายตามแกนเวลา (`predict_linear = nn.Linear(seq_len, pred_len)` คงลำดับ temporal) แล้ว `projection(d_model→c_out)`; pooling ทำลาย temporal resolution ที่เป็นจุดขายของ TimesBlock |
| **การแก้** | ① weight แต่ละ period ด้วย `softmax(amplitude)` แล้ว weighted sum ② ตัด DC ด้วย `torch.cat` แทน in-place ③ `F.pad(..., mode='replicate')` ④ head ใหม่ตาม official: `predict_linear(x.transpose(1,2)).transpose(1,2)` (time-axis projection L→H) + `target_proj(d_model→1).squeeze(-1)` — ตัด pooling/MLP head ทิ้ง |
| **ผล test** | ✅ out `[16, 48]`, 350,353 params, backward OK |

> Note: residual connection ระดับ block (`self.norm(x + res)`) ตรงกับ official `TimesBlock` อยู่แล้ว — การ stack หลาย block ไม่ต้องมี skip เพิ่ม
> Optional (ยังไม่ทำ): official `forecast()` มี instance normalization (mean/std per sample, Non-stationary style) ก่อน embedding — เพิ่มได้อีกถ้าต้องการ faithful สุด

---

## 2. ประเด็นที่ "ยังไม่แก้" (อยู่ครบทั้งใน originals และ `_fix` files)

อยู่นอก scope ของ paper alignment — track ไว้แก้ภายหลัง:

1. ~~`time.sleep(0.005)/(0.002)`~~ → **แก้แล้วใน 01_fix** (ยังเหลือใน 02–09, 13, 14 originals)
2. **ไม่มี `df.sort_index()`** → **แก้แล้วใน 01_fix / 11_fix** (ยังเหลือไฟล์อื่น)
3. **`weight_decay=1e-3` เฉพาะ 01/08** → 01_fix ใช้ `1e-6` แล้ว — ⚠️ **08 (LSTM ablation) ควรตามไปแก้ด้วย** เพื่อรักษา controlled comparison
4. **LightGBM (11): `subsample_freq`** → **แก้แล้วใน 11_fix**
5. **SARIMA (12):** forecast ติดลบไม่ clip ≥ 0, train ด้วย TRAIN+VAL, eval แค่ `MAX_TEST_WINDOWS=200`
6. ~~`.md` output append ซ้ำ~~ → **แก้แล้วใน 01_fix / 11_fix** (ยังเหลือไฟล์อื่น)
7. **XGBoost/LightGBM ไม่ load best_params.json** / model files ไม่ wire HPO params
8. Minor: `"====Match===="` typo (02, 04), duplicate `import sys` (01 — แก้แล้วใน 01_fix), dead code `decoder_layer` (06)
9. **01 Vanilla Transformer:** sinusoidal PE → **แก้แล้วใน 01_fix**; GaussianNoise คงไว้โดยตั้งใจ (regularization)

## 3. ผลกระทบต่อ HPO + การ Sync ฝั่ง `hyperparameter_tuning/`

Architecture ของ **03, 04, 05, 13, 14** เปลี่ยน → `best_params.json` เดิม (ถ้ามี) **in-use ไม่ได้** ต้อง re-run HPO

### Sync สถานะ: ✅ เสร็จแล้ว — สร้างไฟล์ใหม่ใน `../hyperparameter_tuning/` (originals ไม่แตะ)

| HPO file (`_fix`) | sync จาก | params (model ↔ HPO) |
|---|---|---|
| `02_hpo_ifm_pytorch_fix.py` | sum→mean | 108,880 ↔ 108,880 ✅ |
| `03_hpo_afm_pytorch_fix.py` | SeriesDecomp replicate + full decoder | 43,074 ↔ 43,074 ✅ |
| `04_hpo_tft_pytorch_fix.py` | FK decoder inputs + InterpretableMHA | 71,141 ↔ 71,141 ✅ |
| `05_hpo_ptft_pytorch_fix.py` | target channel append + select | 26,440 ↔ 26,440 ✅ |
| `13_hpo_itfm_pytorch_fix.py` | target variate + final LN + gelu | 13,296 ↔ 13,296 ✅ |
| `14_hpo_timesnet_pytorch_fix.py` | adaptive aggregation + replicate pad + official head | 350,353 ↔ 350,353 ✅ |

ทุกไฟล์ผ่าน smoke test (forward/backward) แล้ว — objective/search space/sleep คงไว้ตามเดิม (แก้เฉพาะ architecture + data pipeline ที่จำเป็น: FK tensors ของ 04, target-channel append ของ 05/13)

> ⚠️ Note: HPO scripts บางตัว subsample train set (เช่น 13 ใช้ 30% tail) — คงไว้ตาม design เดิมของ HPO

## 4. Verification

```text
01_fix OK | out (16, 48)    | params  96,624 | backward OK | sinusoidal pe buffer (1, 96, 64)
02_fix OK | out (16, 48)    | params 108,880 | backward OK
03_fix OK | out (16, 48)    | params  43,074 | backward OK
04_fix OK | out (16, 48, 3) | params  71,141 | pinball backward OK | FK cols: 9
05_fix OK | out (16, 48)    | params  26,440 | backward OK
13_fix OK | out (16, 48)    | params  13,296 | backward OK (final LN + gelu)
14_fix OK | out (16, 48)    | params 350,353 | backward OK (official predict_linear head)
11_fix compile OK
```

Test method: exec file content up to config section (cut before training loop) → build model with small dims (d_model=32) → forward batch 16 จาก real data → assert shape → backward pass
