# CNN'lerde Nümerik Optimizasyon: Yakınsama ve Kararlılık Analizi

SGD, Momentum (0.9), RMSprop ve Adam optimizer'larının sığ (3 konv.) ve
BatchNorm'suz derin (8 konv.) CNN'lerde FashionMNIST üzerindeki yakınsama,
seed kararlılığı ve öğrenme oranına bağlı nümerik patlama davranışı.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
python experiments.py            # Deney 1 + Deney 2 + grafikler
python experiments.py --quick    # Birkaç dakikalık duman testi (küçük veri alt kümesi)
python experiments.py --exp 2    # Yalnızca Deney 2
python plot_results.py           # Yalnızca grafikleri yeniden üret
python models.py                 # Model mimarilerini kontrol et
```

Deney 1 tam haliyle 2 model × 4 optimizer × 5 seed × 10 epoch = 400 epoch
eğitir; CPU'da saatler, GPU'da dakikalar sürebilir.

## Çıktılar

| Dosya | İçerik |
|---|---|
| `results/exp1_summary.csv` | Model × optimizer: son kayıp, doğruluk, süre, ilk/son katman gradyan normu (ortalama ± std), %95 GA, CV |
| `results/exp1_epochs.csv` | Epoch bazlı ham ölçümler |
| `results/exp1_layer_grads.csv` | Tüm katmanların epoch bazlı L2 gradyan normları |
| `results/exp1_runs.csv` | Koşu (seed) bazlı son değerler |
| `results/exp2_lr_stability.csv` | lr × optimizer: `stable` / `nan_inf` / `stuck`, patlama epoch'u ve iterasyonu |
| `plots/*.png` | 300 DPI grafikler |

## Metodoloji notları

- **Determinizm:** `set_seed()` → `random`, `numpy`, `torch` (CPU+CUDA) seed'leri,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms`.
  Aynı seed ile tüm optimizer'lar aynı başlangıç ağırlıklarını ve aynı mini-batch
  sırasını görür; farklar yalnızca optimizer'dan kaynaklanır.
- **Gradyan normu:** Her mini-batch'te her ağırlık katmanının `‖∇W‖₂` (Frobenius)
  normu ölçülür, epoch boyunca ortalaması alınır.
- **Patlama tespiti:** Kayıp veya herhangi bir gradyan normu NaN/Inf olduğunda
  koşu durdurulur ve epoch/iterasyon kaydedilir. Sonlu kalıp doğruluğu < %15
  (şans düzeyi ≈ %10) olan koşular `stuck` olarak işaretlenir.
- **Deney 1 lr değerleri:** SGD 0.01, Momentum 0.01, RMSprop 0.001, Adam 0.001.
- **Deney 2:** DeepCNN, seed 42, varsayılan 3 epoch (`--exp2-epochs`).
