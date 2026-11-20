# CNN'lerde Nümerik Optimizasyon: Yakınsama ve Kararlılık Analizi

MAT353 Nümerik Analiz dönem projesi. SGD, Momentum (0.9), RMSprop ve Adam
optimizasyon algoritmaları formüllerinden sıfırdan yazılmış ve sığ (3 konv.)
ile BatchNorm'suz derin (8 konv.) CNN'lerde Fashion-MNIST üzerinde
yakınsama, seed kararlılığı, öğrenme oranı duyarlılığı ve nümerik patlama
bakımından karşılaştırılmıştır.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `MAT353_Proje.ipynb` | Teslim edilecek Colab defteri (testler, canlı demo, tüm deneyler, tablolar, grafikler) |
| `optimizers.py` | SGD / Momentum / RMSprop / Adam: formüllerden sıfırdan uygulama |
| `models.py` | `ShallowCNN` (3 konv.), `DeepCNN` (8 konv., BatchNorm yok) |
| `gradient_check.py` | Otomatik türev vs merkezi sonlu fark, hata analizi |
| `experiments.py` | Deney 1–4 |
| `plot_results.py` | 300 DPI grafikler |
| `tests.py` | Birim testleri |

## Çalıştırma

```bash
pip install -r requirements.txt   # Colab'da gerekmez
python -m unittest tests -v       # birim testleri (~1 dk, veri indirmez)
python experiments.py --quick     # duman testi (~2 dk)
python experiments.py             # tüm deneyler + grafikler
python experiments.py --exp 3 4   # yalnızca sonlu fark ve CPU/GPU deneyleri
python experiments.py --resummarize  # mevcut CSV'lerden özet tabloyu yeniden üret
python plot_results.py            # yalnızca grafikler
```

Fashion-MNIST [Xiao vd., 2017, arXiv:1708.07747] ilk çalıştırmada
`torchvision` tarafından `data/` klasörüne otomatik indirilir.

## Deneyler ve çıktılar

| Deney | İçerik | Çıktılar |
|---|---|---|
| 1 | 2 model × 4 optimizer × 5 seed × 10 epoch: kayıp, doğruluk, süre, katman bazlı ‖∇W‖₂, %85 doğruluğa ulaşma iterasyonu | `exp1_summary.csv` (ort. ± std, %95 GA, CV), `exp1_epochs.csv`, `exp1_layer_grads.csv`, `exp1_runs.csv`; `loss_convergence.png`, `gradient_norm.png`, `time_vs_acc.png` |
| 2 | DeepCNN, 4 optimizer × lr ∈ {0.5, 0.1, 0.01, 0.001, 0.0001}: `stable` / `nan_inf` / `stuck` | `exp2_lr_stability.csv`, `exp2_epochs.csv`; `lr_heatmap.png` |
| 3 | Merkezi sonlu fark göreli hatası vs h, float32 / float64 | `exp3_fd_error.csv`; `fd_error.png` |
| 4 | İterasyon başına süre, CPU ve GPU | `exp4_cpu_gpu.csv`; `cpu_gpu.png` |

## Metodoloji notları

- **Determinizm:** `set_seed()` → `random`, `numpy`, `torch` (CPU+CUDA),
  `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `torch.use_deterministic_algorithms`. Aynı seed ile tüm optimizer'lar aynı
  başlangıç ağırlıklarını ve aynı mini-batch sırasını görür.
- **Optimizer doğrulaması:** `tests.py` sıfırdan uygulamaları elle hesaplanan
  değerlerle ve PyTorch referansıyla (`--optim-impl torch`) karşılaştırır.
- **Patlama tespiti:** Kayıp veya bir gradyan normu NaN/Inf olursa koşu durur,
  epoch/iterasyon kaydedilir. Sonlu kalıp doğruluğu < %15 olanlar `stuck`.
- **Öğrenme oranları (Deney 1):** SGD 0.01, Momentum 0.01, RMSprop 0.001, Adam 0.001.
- **Karmaşıklık:** Bir optimizer adımı O(P) zaman (P = parametre sayısı);
  ek bellek SGD O(1), Momentum/RMSprop O(P), Adam O(2P).
