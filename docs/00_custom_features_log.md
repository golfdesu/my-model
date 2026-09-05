# 00 Custom Model Feature Registry & Roadmap (`00_tfm_custom_pytorch.py`)

เอกสารฉบับนี้เป็นบันทึกรายละเอียดการปรับแต่ง (Customization Registry) ของโมเดล [**`00_tfm_custom_pytorch.py`**](file:///C:/Users/chaya/Documents/Program/Practice/model/00_tfm_custom_pytorch.py) ซึ่งพัฒนาต่อยอดมาจากโมเดลมาตรฐาน [**`01_tfm_tfm_pytorch.py`**](file:///C:/Users/chaya/Documents/Program/Practice/model/01_tfm_tfm_pytorch.py) (Vanilla Transformer - Vaswani et al., 2017) สำหรับพยากรณ์ EV Charging Load ($L=96, H=48$)

---

## 📌 1. สถานะปัจจุบัน (Active Configuration)

โมเดล 00 ปัจจุบันมีโครงสร้างเหมือนโมเดล 01 ทุกประการ (Ablation Baseline) โดยเปิดใช้งานฟีเจอร์ Custom เพียง **1 อย่าง**:

| รายการ | การตั้งค่าปัจจุบันใน `00_tfm_custom_pytorch.py` | เปรียบเทียบกับโมเดล 01 |
| :--- | :--- | :--- |
| **Active Custom Feature** | **Attention Weight Orthogonal Regularization** ($\lambda = 4.5727 \times 10^{-6}$, จูนได้จาก 1D Optuna HPO) | 01 ไม่มี (เพิ่มเข้ามาเฉพาะใน 00) |
| **Positional Encoding** | Fixed Sinusoidal (Vaswani et al., 2017) | เหมือน 01 (100%) |
| **Input Noise** | None (ไม่มีการใส่ Gaussian Noise) | เหมือน 01 (100%) |
| **Architecture Topology** | $d_{\text{model}}=128$, $\text{heads}=4$, $d_{\text{ff}}=256$, $\text{layers}=1$, $\text{dropout}=0.1$ | เหมือน 01 (100%) |
| **Output Head** | 2-Layer MLP: $\text{Concat}(\text{last}, \text{avg}) \to 128 \to 64 \to H=48$ | เหมือน 01 (100%) |
| **Optimization** | $\text{lr}=6.41 \times 10^{-4}$, $\text{weight\_decay}=4.71 \times 10^{-5}$, $\text{batch\_size}=128$ | เหมือน 01 (100%) |

---

## 🎯 2. รายละเอียด Active Feature 1: Attention Orthogonal Regularization

### ทฤษฎีและที่มา
ในโมเดล Attention มักเกิดปัญหา **Attention Collapse** หรือ **Numerical Instability** เมื่อ Weight Matrix $W$ มี Condition Number สูงลิ่ว ($\kappa(W) = \frac{\sigma_{\max}}{\sigma_{\min}} \gg 1000$) การเพิ่ม Orthogonal Penalty จะบังคับให้ $W^T W \approx I$ ทำให้ Singular Values กระจายตัวสม่ำเสมอ

### สมการคณิตศาสตร์
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{ortho}} \sum_{W \in \{W_Q, W_K, W_V, W_O\}} \|W^T W - I\|_F^2$$

### การทำงานในโค้ด
```python
def compute_orthogonal_penalty(model, strength=1e-4):
    if strength <= 0.0:
        return torch.tensor(0.0, device=device)
    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if ('in_proj_weight' in name or 'out_proj.weight' in name) and param.ndim == 2:
            wt_w = torch.matmul(param.t(), param)
            identity = torch.eye(wt_w.size(0), device=param.device)
            penalty = penalty + torch.sum((wt_w - identity) ** 2)
    return strength * penalty
```

---

## 🗂️ 3. เมนู Customization Backlog (พร้อมเปิดใช้งานในอนาคต)

เมื่อต้องการทดลองเปิดฟีเจอร์ใดเพิ่มเติม สามารถระบุชื่อฟีเจอร์เพื่อให้ Agent นำโค้ดส่วนนี้ไปประกอบใน `00_tfm_custom_pytorch.py` ได้ทันที:

### 🔹 Feature 2: Input Gaussian Noise Jittering
* **วัตถุประสงค์:** ทำ Data Perturbation ในมิติ Embedding ป้องกันไม่ให้โมเดลจำ Noise ของเซ็นเซอร์
* **ตำแหน่งโค้ด:**
  ```python
  class GaussianNoise(nn.Module):
      def __init__(self, stddev=0.05):
          super().__init__()
          self.stddev = stddev
      def forward(self, x):
          if self.training and self.stddev > 0.0:
              return x + torch.randn_like(x) * self.stddev
          return x
  ```
* **ค่าเริ่มต้นที่แนะนำ:** `stddev = 0.05`

### 🔹 Feature 3: Trainable Learned Positional Embedding
* **วัตถุประสงค์:** ให้โมเดลเรียนรู้ Temporal Coordinates จากข้อมูลจริงแทนสมมติฐาน Sinusoid ตารางตายตัว
* **ตำแหน่งโค้ด:**
  ```python
  class LearnedPositionalEmbedding(nn.Module):
      def __init__(self, seq_len, d_model):
          super().__init__()
          self.pos_emb = nn.Embedding(seq_len, d_model)
      def forward(self, x):
          positions = torch.arange(0, x.size(1), device=x.device)
          return x + self.pos_emb(positions).unsqueeze(0)
  ```

### 🔹 Feature 4: Compact 1-Layer Projection Head
* **วัตถุประสงค์:** ลด Degree of Freedom ของชั้น Projection Head ป้องกัน Head Overfitting
* **สถาปัตยกรรม:** ปรับจาก 2-layer (`2*d_model -> 128 -> 64 -> H`) เหลือ 1-layer (`2*d_model -> 64 -> H`)
* **ตำแหน่งโค้ด:**
  ```python
  self.head_fc = nn.Linear(d_model * 2, 64)
  self.head_dropout = nn.Dropout(dropout_rate)
  self.out_proj = nn.Linear(64, horizon)
  ```

### 🔹 Feature 5: Deeper & Thinner Topology
* **วัตถุประสงค์:** เพิ่มความลึกของ Representation Hierarchy โดยไม่เพิ่ม Parameter Budget
* **การตั้งค่า:**
  * `NUM_LAYERS = 2` (จากเดิม 1)
  * `D_MODEL = 64` (จากเดิม 128)
  * `D_FF = 128` (จากเดิม 256)

### 🔹 Feature 6: Aggressive Regularization Suite
* **วัตถุประสงค์:** คุม Generalization สำหรับข้อมูลที่มีความผันผวนสูง
* **การตั้งค่า:**
  * `DROPOUT_RATE = 0.2` (จากเดิม 0.1)
  * `WEIGHT_DECAY = 1e-3` (L2 Regularization เพิ่มขึ้นประมาณ 20 เท่า)

### 🔹 Feature 7: Smaller Batch Size Optimization
* **วัตถุประสงค์:** อาศัย Stochastic Gradient Noise ช่วยหลุดจาก Saddle Points / Local Minima ที่แหลมเกินไป
* **การตั้งค่า:** `BATCH_SIZE = 64` (จากเดิม 128)

### 🔹 Feature 8: Peak-Weighted Loss Function
* **วัตถุประสงค์:** ป้องกันการทำนายค่า Peak ต่ำเกินไป (Peak Underprediction) โดยให้น้ำหนัก Loss เพิ่มขึ้น $5\times$ ในช่วงโหลดเกิน $P_{80}$ ของ Train set
* **สมการ Loss:**
  $$\mathcal{L}_{\text{weighted}} = \frac{1}{B \cdot H} \sum_{i, t} w_{i, t} (y_{i, t} - \hat{y}_{i, t})^2, \quad w_{i, t} = \begin{cases} 5.0 & \text{if } y_{i, t} \ge P_{80} \\ 1.0 & \text{otherwise} \end{cases}$$

---

## 📈 4. บันทึกผลการทดลอง (Experiment Log)

| รหัสการทดลอง | วันที่ | สิ่งที่ Custom เพิ่มเติม | MAE รวม | RMSE รวม | $R^2$ | Peak WAPE | หมายเหตุ |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **01 (Baseline)** | Benchmark | Vanilla Transformer (ไม่มี Custom) | - | - | - | - | ค่ามาตรฐาน Vaswani 2017 (Val Loss = 0.003087) |
| **00 (HPO)** | 2026-09-05 | 1D Optuna Search (50 trials, 30 epochs) -> $\lambda^* = 4.5727 \times 10^{-6}$ | Val Loss = 0.003006 | - | - | - | ชนะโมเดล 01 (Trial 25, Val Loss ดีกว่า 01) |
| **00 (Benchmark)** | 2026-09-05 | 10-Seed Benchmark ด้วย $\lambda^* = 4.5727 \times 10^{-6}$ | *พร้อมรัน* | *พร้อมรัน* | *พร้อมรัน* | *พร้อมรัน* | สั่งรันผ่าน run_benchmark_00.sbatch บน H100 |
