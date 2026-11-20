"""
experiments.py
==============
CNN'lerde nümerik optimizasyon algoritmalarının (SGD, Momentum, RMSprop, Adam)
derinlik ve gradyan kararsızlığı koşullarında yakınsama/kararlılık analizi.

Deney 1 - Yakınsama ve seed kararlılığı
    ShallowCNN ve DeepCNN, 4 optimizer x 5 seed x 10 epoch.
    Kaydedilenler: epoch başına eğitim kaybı, test doğruluğu, süre (s) ve
    ilk/son katman L2 gradyan normları ||grad||_2 (+ tüm katmanlar).
    -> results/exp1_epochs.csv        (ham, epoch bazlı)
    -> results/exp1_layer_grads.csv   (katman bazlı gradyan normları)
    -> results/exp1_runs.csv          (koşu başına son değerler)
    -> results/exp1_summary.csv       (ortalama ± standart sapma)

Deney 2 - Öğrenme oranı hassasiyeti ve nümerik patlama
    DeepCNN, 4 optimizer x lr = [0.5, 0.1, 0.01, 0.001, 0.0001].
    NaN / Inf oluşan (taşma/patlama) konfigürasyonlar tespit edilir.
    -> results/exp2_lr_stability.csv
    -> results/exp2_epochs.csv

Deney 3 - Gradyan doğrulaması ve sonlu fark hata analizi
    Otomatik türev gradyanı merkezi sonlu farkla karşılaştırılır; h'ye göre
    kesme (O(h²)) ve yuvarlama (O(ε/h)) hataları float32/float64 için ölçülür.
    -> results/exp3_fd_error.csv

Deney 4 - CPU / GPU performans karşılaştırması
    İterasyon başına süre (ileri + geri + optimizer adımı).
    -> results/exp4_cpu_gpu.csv

Optimizer'lar optimizers.py içinde formüllerinden sıfırdan yazılmıştır
(--optim-impl manual, varsayılan). --optim-impl torch ile PyTorch'un referans
uygulamaları kullanılır; ikisinin eşdeğerliği tests.py ile doğrulanır.

Kullanım:
    python experiments.py                 # tüm deneyler + grafikler
    python experiments.py --exp 1         # yalnızca Deney 1
    python experiments.py --quick         # küçük veri alt kümesiyle hızlı duman testi
    python experiments.py --resummarize   # mevcut CSV'lerden exp1_summary.csv'yi yeniden üret
"""

import os

# CUDA üzerinde deterministik cuBLAS için torch import edilmeden ÖNCE ayarlanmalı
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torchvision import datasets

from gradient_check import default_hs, directional_fd_errors
from models import build_model, weight_layers
from optimizers import make_optimizer

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
SEEDS = [42, 100, 2024, 7, 99]
MODELS = ["ShallowCNN", "DeepCNN"]
OPTIMIZERS = ["SGD", "Momentum", "RMSprop", "Adam"]
LR_GRID = [0.5, 0.1, 0.01, 0.001, 0.0001]

# Deney 1'de her optimizer için literatürde yaygın varsayılan öğrenme oranları
DEFAULT_LR = {"SGD": 0.01, "Momentum": 0.01, "RMSprop": 0.001, "Adam": 0.001}

# FashionMNIST eğitim kümesi ortalama / standart sapması (normalizasyon için)
FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530

# Şans düzeyi 10 sınıf için %10; altında kalan sonlu koşular "takıldı" sayılır
STUCK_ACC_THRESHOLD = 0.15

# Yakınsama hızı ölçütü: test doğruluğunun ilk kez bu eşiğe ulaştığı epoch/iterasyon
TARGET_ACC = 0.85


# ---------------------------------------------------------------------------
# Tekrarlanabilirlik
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Tüm rastgelelik kaynaklarını sabitler (deterministik sonuçlar için)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Deterministik karşılığı olmayan bir işlem varsa hata yerine uyarı ver
    torch.use_deterministic_algorithms(True, warn_only=True)


def sync(device: torch.device) -> None:
    """GPU'da süre ölçümünün doğru olması için asenkron çekirdekleri bekler."""
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------
def load_fashion_mnist(device, data_dir="data", train_subset=None, test_subset=None):
    """
    FashionMNIST'i indirir ve tamamını normalize edilmiş tensörler olarak
    belleğe (gerekirse GPU'ya) yükler. DataLoader yerine bellek içi indeksleme
    kullanmak hem hızlıdır hem de karıştırma sırasını tam kontrol etmemizi sağlar.
    """
    train = datasets.FashionMNIST(data_dir, train=True, download=True)
    test = datasets.FashionMNIST(data_dir, train=False, download=True)

    def prep(ds, n):
        X = ds.data.float().div(255.0).unsqueeze(1)          # (N, 1, 28, 28)
        X = (X - FMNIST_MEAN) / FMNIST_STD
        y = ds.targets.clone().long()
        if n:
            X, y = X[:n], y[:n]
        return X.to(device), y.to(device)

    Xtr, ytr = prep(train, train_subset)
    Xte, yte = prep(test, test_subset)
    print(f"Veri yüklendi: eğitim {tuple(Xtr.shape)}, test {tuple(Xte.shape)}, cihaz={device}")
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# Eğitim / değerlendirme
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, layers, X, y, batch_size, generator):
    """
    Bir epoch eğitir. Her mini-batch'te:
      * kayıp sonlu mu (NaN/Inf) kontrol edilir,
      * her ağırlık katmanının gradyanının L2 (Frobenius) normu ||grad||_2 ölçülür.
    Dönen sözlük: ortalama kayıp, katman bazlı ortalama gradyan normları,
    epoch içindeki en büyük toplam gradyan normu ve patlama bilgisi.
    """
    model.train()
    n = X.shape[0]
    perm = torch.randperm(n, generator=generator).to(X.device)

    loss_sum, n_batches = 0.0, 0
    grad_sums = np.zeros(len(layers), dtype=np.float64)
    max_total_grad = 0.0

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        xb, yb = X[idx], y[idx]

        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss_val = loss.item()

        # İleri geçişte taşma: kayıp NaN veya Inf
        if not math.isfinite(loss_val):
            return dict(diverged=True, diverged_iter=n_batches + 1, train_loss=float("nan"),
                        grad_norms=np.full(len(layers), np.nan), max_grad=float("inf"))

        loss.backward()

        # Katman bazlı gradyan L2 normları (float64'e çevirip toplam normu hesapla)
        norms = torch.stack([m.weight.grad.detach().norm(2) for _, m in layers])
        norms = norms.double().cpu().numpy()
        total = float(np.sqrt(np.sum(norms ** 2)))

        # Geri yayılımda taşma: gradyan NaN veya Inf
        if not np.all(np.isfinite(norms)):
            return dict(diverged=True, diverged_iter=n_batches + 1, train_loss=float("nan"),
                        grad_norms=np.full(len(layers), np.nan), max_grad=float("inf"))

        optimizer.step()

        loss_sum += loss_val
        grad_sums += norms
        max_total_grad = max(max_total_grad, total)
        n_batches += 1

    return dict(diverged=False, diverged_iter=None, train_loss=loss_sum / n_batches,
                grad_norms=grad_sums / n_batches, max_grad=max_total_grad)


@torch.no_grad()
def evaluate(model, X, y, batch_size=1000):
    """Test doğruluğu ve ortalama test kaybını döndürür."""
    model.eval()
    n, correct, loss_sum = X.shape[0], 0, 0.0
    for s in range(0, n, batch_size):
        out = model(X[s:s + batch_size])
        loss_sum += F.cross_entropy(out, y[s:s + batch_size], reduction="sum").item()
        correct += (out.argmax(1) == y[s:s + batch_size]).sum().item()
    return correct / n, loss_sum / n


def run_training(model_name, opt_name, lr, seed, epochs, data, batch_size, device, tag="",
                 impl="manual"):
    """
    Tek bir (model, optimizer, lr, seed) koşusunu çalıştırır.
    NaN/Inf tespit edilirse eğitim o noktada durdurulur.
    """
    Xtr, ytr, Xte, yte = data
    iters_per_epoch = math.ceil(Xtr.shape[0] / batch_size)

    # Aynı seed -> aynı başlangıç ağırlıkları ve aynı mini-batch sırası.
    # Böylece optimizer'lar birebir aynı başlangıç koşullarında karşılaştırılır.
    set_seed(seed)
    model = build_model(model_name).to(device)
    layers = weight_layers(model)
    optimizer = make_optimizer(opt_name, model.parameters(), lr, impl=impl)
    gen = torch.Generator().manual_seed(seed)

    epoch_rows, layer_rows = [], []
    status, div_epoch, div_iter = "stable", None, None
    cum_time, max_grad = 0.0, 0.0
    base = dict(model=model_name, optimizer=opt_name, lr=lr, seed=seed)

    for epoch in range(1, epochs + 1):
        sync(device)
        t0 = time.perf_counter()
        r = train_one_epoch(model, optimizer, layers, Xtr, ytr, batch_size, gen)
        sync(device)
        dt = time.perf_counter() - t0
        cum_time += dt
        max_grad = max(max_grad, r["max_grad"])

        if r["diverged"]:
            status, div_epoch, div_iter = "nan_inf", epoch, r["diverged_iter"]
            epoch_rows.append(dict(base, epoch=epoch,
                                   iteration=(epoch - 1) * iters_per_epoch + div_iter,
                                   train_loss=np.nan, test_loss=np.nan,
                                   test_acc=np.nan, epoch_time=dt, cum_time=cum_time,
                                   grad_first=np.nan, grad_last=np.nan))
            print(f"  {tag} epoch {epoch:2d}: NaN/Inf tespit edildi (iterasyon {div_iter}) -> durduruldu")
            break

        test_acc, test_loss = evaluate(model, Xte, yte)
        # Son optimizer adımı ağırlıkları bozmuş olabilir (test kaybı sonlu değil)
        if not math.isfinite(test_loss):
            status, div_epoch = "nan_inf", epoch

        epoch_rows.append(dict(base, epoch=epoch, iteration=epoch * iters_per_epoch,
                               train_loss=r["train_loss"], test_loss=test_loss,
                               test_acc=test_acc, epoch_time=dt, cum_time=cum_time,
                               grad_first=r["grad_norms"][0], grad_last=r["grad_norms"][-1]))
        for li, ((lname, _), g) in enumerate(zip(layers, r["grad_norms"])):
            layer_rows.append(dict(base, epoch=epoch, layer_idx=li, layer_name=lname, grad_norm=g))

        print(f"  {tag} epoch {epoch:2d}: loss={r['train_loss']:.4f}  acc={test_acc * 100:5.2f}%  "
              f"|g1|={r['grad_norms'][0]:.2e}  |gL|={r['grad_norms'][-1]:.2e}  t={dt:.1f}s")

        if status == "nan_inf":
            print(f"  {tag} test kaybı sonlu değil -> durduruldu")
            epoch_rows[-1]["test_acc"] = np.nan
            break

    final = epoch_rows[-1]
    if status == "stable" and final["test_acc"] < STUCK_ACC_THRESHOLD:
        status = "stuck"  # sayısal olarak sonlu ama öğrenemedi (şans düzeyi)

    summary = dict(base, status=status, diverged_epoch=div_epoch, diverged_iter=div_iter,
                   epochs_completed=len(epoch_rows), final_train_loss=final["train_loss"],
                   final_test_acc=final["test_acc"],
                   best_test_acc=np.nanmax([e["test_acc"] for e in epoch_rows])
                   if any(np.isfinite(e["test_acc"]) for e in epoch_rows) else np.nan,
                   iters_per_epoch=iters_per_epoch,
                   total_time=cum_time, max_grad_norm=max_grad,
                   final_grad_first=final["grad_first"], final_grad_last=final["grad_last"])
    return epoch_rows, layer_rows, summary


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def fmt_pm(values, fmt="{:.4f}"):
    """Bir dizi değeri 'ortalama ± std' metnine çevirir (örnek std, ddof=1)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "NaN"
    std = v.std(ddof=1) if v.size > 1 else 0.0
    return f"{fmt.format(v.mean())} ± {fmt.format(std)}"


def ci95_halfwidth(values):
    """Student-t dağılımı ile %95 güven aralığı yarı genişliği."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    return stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)


def epochs_to_target(epochs_df: pd.DataFrame, target=TARGET_ACC) -> pd.DataFrame:
    """Her koşu için test doğruluğunun ilk kez `target`'a ulaştığı epoch (yoksa NaN)."""
    out = []
    for (model, opt, seed), g in epochs_df.groupby(["model", "optimizer", "seed"]):
        hit = g[g.test_acc >= target].epoch
        out.append(dict(model=model, optimizer=opt, seed=seed,
                        epochs_to_target=hit.min() if len(hit) else np.nan))
    return pd.DataFrame(out)


def summarize_exp1(runs_df: pd.DataFrame, epochs_df: pd.DataFrame,
                   iters_per_epoch: int) -> pd.DataFrame:
    """Koşu bazlı sonuçları (model, optimizer) bazında ortalama ± std'ye indirger."""
    ett = epochs_to_target(epochs_df)
    rows = []
    for model in MODELS:
        for opt in OPTIMIZERS:
            g = runs_df[(runs_df.model == model) & (runs_df.optimizer == opt)]
            if g.empty:
                continue
            acc = g.final_test_acc * 100
            acc_f = acc[np.isfinite(acc)]
            e = ett[(ett.model == model) & (ett.optimizer == opt)].epochs_to_target
            n_hit = int(np.isfinite(e).sum())
            rows.append({
                "model": model,
                "optimizer": opt,
                "lr": g.lr.iloc[0],
                "n_seeds": len(g),
                "n_diverged": int((g.status == "nan_inf").sum()),
                f"reached_{int(TARGET_ACC * 100)}%": f"{n_hit}/{len(e)}",
                f"epochs_to_{int(TARGET_ACC * 100)}%": fmt_pm(e, "{:.1f}") if n_hit else "ulaşılamadı",
                f"iters_to_{int(TARGET_ACC * 100)}%": fmt_pm(e * iters_per_epoch, "{:.0f}")
                if n_hit else "ulaşılamadı",
                "final_train_loss": fmt_pm(g.final_train_loss),
                "final_test_acc(%)": fmt_pm(acc, "{:.2f}"),
                "test_acc_95CI(%)": f"± {ci95_halfwidth(acc):.2f}",
                "test_acc_CV(%)": f"{(acc_f.std(ddof=1) / acc_f.mean() * 100):.3f}"
                if len(acc_f) > 1 else "NaN",
                "best_test_acc(%)": fmt_pm(g.best_test_acc * 100, "{:.2f}"),
                "total_time(s)": fmt_pm(g.total_time, "{:.1f}"),
                "grad_norm_first_layer": fmt_pm(g.final_grad_first, "{:.3e}"),
                "grad_norm_last_layer": fmt_pm(g.final_grad_last, "{:.3e}"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deneyler
# ---------------------------------------------------------------------------
def experiment_1(args, data, device):
    print("\n" + "=" * 70)
    print("DENEY 1: Yakınsama ve seed kararlılığı")
    print("=" * 70)
    epoch_rows, layer_rows, run_rows = [], [], []
    total = len(args.models) * len(OPTIMIZERS) * len(args.seeds)
    k = 0
    for model_name in args.models:
        for opt in OPTIMIZERS:
            for seed in args.seeds:
                k += 1
                tag = f"[{k}/{total}] {model_name:<10} {opt:<8} seed={seed:<4}"
                print(tag)
                e, l, s = run_training(model_name, opt, DEFAULT_LR[opt], seed, args.epochs,
                                       data, args.batch_size, device, tag="   ",
                                       impl=args.optim_impl)
                epoch_rows += e
                layer_rows += l
                run_rows.append(s)
                # Her koşudan sonra kaydet: uzun deney yarıda kesilse de veri kaybolmaz
                save_csv(epoch_rows, os.path.join(args.results_dir, "exp1_epochs.csv"))
                save_csv(layer_rows, os.path.join(args.results_dir, "exp1_layer_grads.csv"))
                save_csv(run_rows, os.path.join(args.results_dir, "exp1_runs.csv"))

    write_exp1_summary(pd.DataFrame(run_rows), pd.DataFrame(epoch_rows),
                       math.ceil(data[0].shape[0] / args.batch_size), args.results_dir)


def write_exp1_summary(runs_df, epochs_df, iters_per_epoch, results_dir):
    summary = summarize_exp1(runs_df, epochs_df, iters_per_epoch)
    path = os.path.join(results_dir, "exp1_summary.csv")
    summary.to_csv(path, index=False, encoding="utf-8-sig")  # BOM: Excel'de '±' doğru görünsün
    print("\nDeney 1 özeti:")
    print(summary.to_string(index=False))
    print(f"-> {path}")


def experiment_2(args, data, device):
    print("\n" + "=" * 70)
    print("DENEY 2: Öğrenme oranı hassasiyeti ve nümerik patlama (DeepCNN)")
    print("=" * 70)
    seed = args.seeds[0]
    epoch_rows, rows = [], []
    total = len(OPTIMIZERS) * len(LR_GRID)
    k = 0
    for opt in OPTIMIZERS:
        for lr in LR_GRID:
            k += 1
            tag = f"[{k}/{total}] DeepCNN {opt:<8} lr={lr:<7g}"
            print(tag)
            e, _, s = run_training("DeepCNN", opt, lr, seed, args.exp2_epochs,
                                   data, args.batch_size, device, tag="   ",
                                   impl=args.optim_impl)
            epoch_rows += e
            rows.append(s)
            save_csv(epoch_rows, os.path.join(args.results_dir, "exp2_epochs.csv"))

    df = pd.DataFrame(rows)
    df["nan_or_inf"] = df.status == "nan_inf"
    cols = ["optimizer", "lr", "seed", "status", "nan_or_inf", "diverged_epoch", "diverged_iter",
            "epochs_completed", "final_train_loss", "final_test_acc", "best_test_acc",
            "max_grad_norm", "total_time"]
    df = df[cols]
    path = os.path.join(args.results_dir, "exp2_lr_stability.csv")
    df.to_csv(path, index=False)
    print("\nDeney 2 sonuçları:")
    print(df.to_string(index=False))
    print(f"-> {path}")


def experiment_3(args, data):
    """Otomatik türev vs merkezi sonlu fark: h'ye göre göreli hata (CPU, float32/64)."""
    print("\n" + "=" * 70)
    print("DENEY 3: Gradyan doğrulaması ve sonlu fark hata analizi")
    print("=" * 70)
    _, _, Xte, yte = data
    rows = []
    for model_name in MODELS:
        r = directional_fd_errors(model_name, Xte, yte, default_hs(), seed=args.seeds[0])
        rows += r
        for dtype in ("float64", "float32"):
            sub = [x for x in r if x["dtype"] == dtype]
            best = min(sub, key=lambda x: x["rel_error"])
            print(f"  {model_name:<10} {dtype}: en küçük göreli hata {best['rel_error']:.2e} "
                  f"(h = {best['h']:.1e})")
    path = os.path.join(args.results_dir, "exp3_fd_error.csv")
    save_csv(rows, path)
    print(f"-> {path}")


def experiment_4(args):
    """İterasyon başına süre: CPU ve (varsa) GPU. Rastgele veriyle, ManualAdam ile."""
    print("\n" + "=" * 70)
    print("DENEY 4: CPU / GPU performans karşılaştırması")
    print("=" * 70)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("  CUDA bulunamadı: yalnızca CPU ölçülecek.")

    rows = []
    n_warmup, n_iters = 5, args.bench_iters
    for device in devices:
        name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        for model_name in MODELS:
            set_seed(args.seeds[0])
            model = build_model(model_name).to(device)
            opt = make_optimizer("Adam", model.parameters(), 1e-3, impl=args.optim_impl)
            xb = torch.randn(args.batch_size, 1, 28, 28, device=device)
            yb = torch.randint(0, 10, (args.batch_size,), device=device)

            times = []
            for i in range(n_warmup + n_iters):
                sync(device)
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
                sync(device)
                if i >= n_warmup:  # ısınma iterasyonları (bellek ayırma, çekirdek seçimi) sayılmaz
                    times.append((time.perf_counter() - t0) * 1000)
            t = np.array(times)
            rows.append(dict(device=device.type, device_name=name, model=model_name,
                             batch_size=args.batch_size, n_iters=n_iters,
                             ms_per_iter_mean=t.mean(), ms_per_iter_std=t.std(ddof=1),
                             iters_per_sec=1000 / t.mean()))
            print(f"  {name:<20} {model_name:<10} {t.mean():8.2f} ± {t.std(ddof=1):.2f} ms/iter")

    df = pd.DataFrame(rows)
    if "cuda" in set(df.device):
        cpu = df[df.device == "cpu"].set_index("model").ms_per_iter_mean
        df["speedup_vs_cpu"] = [cpu[m] / v for m, v in zip(df.model, df.ms_per_iter_mean)]
    path = os.path.join(args.results_dir, "exp4_cpu_gpu.csv")
    df.to_csv(path, index=False)
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CNN optimizer yakınsama/kararlılık deneyleri")
    p.add_argument("--exp", nargs="+", choices=["1", "2", "3", "4", "all"], default=["all"],
                   help="Çalıştırılacak deney(ler), örn. --exp 3 4")
    p.add_argument("--epochs", type=int, default=10, help="Deney 1 epoch sayısı")
    p.add_argument("--exp2-epochs", type=int, default=3, help="Deney 2 epoch sayısı")
    p.add_argument("--bench-iters", type=int, default=50, help="Deney 4 ölçülen iterasyon sayısı")
    p.add_argument("--optim-impl", choices=["manual", "torch"], default="manual",
                   help="manual: optimizers.py (sıfırdan), torch: PyTorch referansı")
    p.add_argument("--resummarize", action="store_true",
                   help="Eğitim yapmadan mevcut exp1 CSV'lerinden exp1_summary.csv'yi yeniden üret")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--device", default="auto", help="'auto', 'cpu' veya 'cuda'")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--quick", action="store_true",
                   help="Duman testi: 6000 eğitim / 2000 test örneği, 2 epoch, 2 seed")
    p.add_argument("--no-plot", action="store_true", help="Sonda grafikleri üretme")
    return p.parse_args()


def main():
    args = parse_args()
    if args.resummarize:
        runs = pd.read_csv(os.path.join(args.results_dir, "exp1_runs.csv"))
        epochs = pd.read_csv(os.path.join(args.results_dir, "exp1_epochs.csv"))
        ipe = (int(runs.iters_per_epoch.iloc[0]) if "iters_per_epoch" in runs
               else math.ceil(60000 / args.batch_size))
        write_exp1_summary(runs, epochs, ipe, args.results_dir)
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_subset = test_subset = None
    if args.quick:
        train_subset, test_subset = 6000, 2000
        args.epochs, args.exp2_epochs = 2, 1
        args.seeds = args.seeds[:2]
        print("HIZLI MOD: sonuçlar yalnızca kodun çalıştığını doğrulamak içindir.")

    os.makedirs(args.results_dir, exist_ok=True)
    data = load_fashion_mnist(device, args.data_dir, train_subset, test_subset)

    exps = {"1", "2", "3", "4"} if "all" in args.exp else set(args.exp)
    print(f"Optimizer uygulaması: {args.optim_impl}")
    t0 = time.perf_counter()
    if "1" in exps:
        experiment_1(args, data, device)
    if "2" in exps:
        experiment_2(args, data, device)
    if "3" in exps:
        experiment_3(args, data)
    if "4" in exps:
        experiment_4(args)
    print(f"\nToplam süre: {(time.perf_counter() - t0) / 60:.1f} dk")

    if not args.no_plot:
        import plot_results
        plot_results.main(results_dir=args.results_dir, plots_dir=args.plots_dir)


if __name__ == "__main__":
    main()
