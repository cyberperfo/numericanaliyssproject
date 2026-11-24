"""
plot_results.py
===============
results/ altındaki CSV dosyalarından 300 DPI akademik grafikler üretir ve
plots/ klasörüne kaydeder:

    loss_convergence.png : Eğitim kaybı vs epoch (log y), ortalama ± std bantları
    gradient_norm.png    : ShallowCNN vs DeepCNN katman bazlı gradyan normları
                           ve ilk/son katman gradyan oranının epoch'larla değişimi
    lr_heatmap.png       : Optimizer x öğrenme oranı kararlılık/doğruluk ısı haritası
    time_vs_acc.png      : Toplam eğitim süresi vs son test doğruluğu
    fd_error.png         : Sonlu fark göreli hatası vs h (kesme/yuvarlama dengesi)
    cpu_gpu.png          : CPU ve GPU'da iterasyon başına süre
    sharpness.png        : Kuvvet yöntemi yakınsaması, λ_max(∇²L) ve η_kritik = 2/λ_max
    ablation.png         : BatchNorm ve Adam yanlılık düzeltmesi ablasyonu

Kullanım:
    python plot_results.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # ekran gerektirmeyen arka uç
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

DPI = 300
OPT_ORDER = ["SGD", "Momentum", "RMSprop", "Adam"]
MODEL_ORDER = ["ShallowCNN", "DeepCNN"]

# Renk körlüğü dostu, sabit sıralı kategorik palet (her optimizer hep aynı renk)
OPT_COLORS = {"SGD": "#2a78d6", "Momentum": "#eb6834", "RMSprop": "#1baf7a", "Adam": "#eda100"}
# İkincil kodlama: renk tek başına kimlik taşımasın diye her optimizer'a ayrı işaretçi
OPT_MARKERS = {"SGD": "o", "Momentum": "s", "RMSprop": "^", "Adam": "D"}
MODEL_STYLE = {
    "ShallowCNN": dict(color="#2a78d6", linestyle="-", marker="o"),
    "DeepCNN": dict(color="#eb6834", linestyle="--", marker="s"),
}

# Nötr mürekkep / ızgara tonları
INK = "#2b2b28"
MUTED = "#6b6a63"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Tek tonlu (mavi) sıralı renk skalası: açık = düşük doğruluk, koyu = yüksek
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def _band(mean, std, floor=1e-12):
    """
    Log ölçekte çizilebilen ± std bandı: [m²/(m+s), m+s].
    Alt sınır, üst sınırın log ölçekte aynası (m/(1+s/m)); s ≪ m iken ≈ m − s olur,
    s ≥ m olduğunda ise m − s ≤ 0'ın yarattığı yapay "dibe inen" bandı önler.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.nan_to_num(np.asarray(std, dtype=float))
    hi = mean + std
    lo = np.maximum(mean * mean / np.maximum(hi, floor), floor)
    return lo, hi


def _save(fig, plots_dir, name):
    path = os.path.join(plots_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  kaydedildi: {path}")


# ---------------------------------------------------------------------------
# 1) Kayıp yakınsaması
# ---------------------------------------------------------------------------
def plot_loss_convergence(epochs_df, plots_dir):
    df = epochs_df[np.isfinite(epochs_df.train_loss)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = df[df.model == model]
        for opt in OPT_ORDER:
            s = sub[sub.optimizer == opt]
            if s.empty:
                continue
            g = s.groupby("epoch")["train_loss"].agg(["mean", "std"]).reset_index()
            lo, hi = _band(g["mean"].values, g["std"].values)
            ax.fill_between(g.epoch, lo, hi, color=OPT_COLORS[opt], alpha=0.18, linewidth=0)
            ax.plot(g.epoch, g["mean"], color=OPT_COLORS[opt], marker=OPT_MARKERS[opt],
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=opt)
        ax.set_yscale("log")
        ax.set_title(model)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if sub.empty:
            ax.text(0.5, 0.5, "veri yok", transform=ax.transAxes, ha="center", color=MUTED)
    axes[0].set_ylabel("Eğitim kaybı (log ölçek)")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.04, "Çizgi: seed'ler üzerinden ortalama; gölge: ± 1 standart sapma",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    _save(fig, plots_dir, "loss_convergence.png")


# ---------------------------------------------------------------------------
# 2) Gradyan normları
# ---------------------------------------------------------------------------
def plot_gradient_norm(layer_df, epochs_df, plots_dir, grad_epoch=1):
    """
    Üst satır : epoch `grad_epoch` boyunca ortalama ||grad||_2, katman sırasına göre
                (girdi -> çıktı). DeepCNN'de ilk katmanlara doğru düşüş = gradyan sönmesi.
    Alt satır : ||g_ilk|| / ||g_son|| oranının epoch'larla değişimi (1'den çok küçük
                ise ilk katmanlar neredeyse öğrenmiyor demektir).
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8))
    for j, opt in enumerate(OPT_ORDER):
        ax_top, ax_bot = axes[0, j], axes[1, j]
        for model in MODEL_ORDER:
            st = MODEL_STYLE[model]

            # --- katman bazlı norm ---
            s = layer_df[(layer_df.model == model) & (layer_df.optimizer == opt)
                         & (layer_df.epoch == grad_epoch) & np.isfinite(layer_df.grad_norm)]
            if not s.empty:
                g = (s.groupby(["layer_idx", "layer_name"])["grad_norm"]
                      .agg(["mean", "std"]).reset_index().sort_values("layer_idx"))
                x = g.layer_idx.values + 1
                lo, hi = _band(g["mean"].values, g["std"].values)
                ax_top.fill_between(x, lo, hi, color=st["color"], alpha=0.15, linewidth=0)
                ax_top.plot(x, g["mean"], color=st["color"], linestyle=st["linestyle"],
                            marker=st["marker"], markersize=4.5, label=model)

            # --- ilk/son katman oranı ---
            e = epochs_df[(epochs_df.model == model) & (epochs_df.optimizer == opt)].copy()
            e["ratio"] = e.grad_first / e.grad_last
            e = e[np.isfinite(e.ratio) & (e.ratio > 0)]
            if not e.empty:
                g = e.groupby("epoch")["ratio"].agg(["mean", "std"]).reset_index()
                lo, hi = _band(g["mean"].values, g["std"].values)
                ax_bot.fill_between(g.epoch, lo, hi, color=st["color"], alpha=0.15, linewidth=0)
                ax_bot.plot(g.epoch, g["mean"], color=st["color"], linestyle=st["linestyle"],
                            marker=st["marker"], markersize=4.5, label=model)

        ax_top.set_yscale("log")
        ax_top.set_title(opt)
        ax_top.set_xlabel("Katman indeksi (girdi → çıktı)")
        ax_top.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        ax_bot.set_yscale("log")
        ax_bot.axhline(1.0, color=AXIS, linestyle=":", linewidth=1.2)
        ax_bot.set_xlabel("Epoch")
        ax_bot.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    axes[0, 0].set_ylabel(f"‖∇W‖₂  (epoch {grad_epoch} ortalaması)")
    axes[1, 0].set_ylabel("‖∇W_ilk‖₂ / ‖∇W_son‖₂")
    handles = [Line2D([0], [0], color=MODEL_STYLE[m]["color"], linestyle=MODEL_STYLE[m]["linestyle"],
                      marker=MODEL_STYLE[m]["marker"], label=m) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.02,
             "Üst: katman bazlı L2 gradyan normu (seed ortalaması ± std). "
             "Alt: ilk/son katman gradyan oranı; 1'in çok altı → gradyan sönmesi.",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    _save(fig, plots_dir, "gradient_norm.png")


# ---------------------------------------------------------------------------
# 3) Öğrenme oranı ısı haritası
# ---------------------------------------------------------------------------
def plot_lr_heatmap(exp2_df, plots_dir):
    lrs = sorted(exp2_df.lr.unique(), reverse=True)
    opts = [o for o in OPT_ORDER if o in set(exp2_df.optimizer)]
    # Birden çok seed varsa hücre = seed'ler üzerinden özet:
    #   renk/metin: sonlu koşuların ortalama doğruluğu; alt satır: başarısız seed sayıları
    acc = np.full((len(opts), len(lrs)), np.nan)
    status = np.full((len(opts), len(lrs)), "", dtype=object)
    div_ep = np.full((len(opts), len(lrs)), np.nan)
    all_nan = np.zeros((len(opts), len(lrs)), dtype=bool)
    n_seeds = exp2_df.groupby(["optimizer", "lr"]).size().max()

    for (opt, lr), g in exp2_df.groupby(["optimizer", "lr"]):
        if opt not in opts:
            continue
        i, j = opts.index(opt), lrs.index(lr)
        a = g.final_test_acc.values * 100
        acc[i, j] = np.nanmean(a) if np.isfinite(a).any() else np.nan
        n = len(g)
        counts = g.status.value_counts()
        all_nan[i, j] = counts.get("nan_inf", 0) == n
        div_ep[i, j] = g.diverged_epoch.min() if g.diverged_epoch.notna().any() else np.nan
        if n == 1:
            status[i, j] = g.status.iloc[0]
        else:
            parts = [f"{lbl} {counts[s]}/{n}" for s, lbl in
                     (("nan_inf", "NaN"), ("exploded", "patladı"), ("stuck", "takıldı"))
                     if counts.get(s, 0) > 0]
            status[i, j] = "\n".join(parts)

    exploded = all_nan | ~np.isfinite(acc)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cmap.set_bad(SURFACE)

    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(opts) + 1.8))
    im = ax.imshow(np.ma.masked_where(exploded, acc), cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(len(opts)):
        for j in range(len(lrs)):
            if exploded[i, j]:
                # Patlayan hücre: taralı, renk skalasının dışında
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=GRID,
                                       edgecolor="#a8a79d", hatch="///", linewidth=0))
                ep = f"\nepoch {int(div_ep[i, j])}" if np.isfinite(div_ep[i, j]) else ""
                tag = f" {n_seeds}/{n_seeds}" if n_seeds > 1 else ""
                ax.text(j, i, f"NaN/Inf{tag}{ep}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color=INK)
            else:
                txt = f"{acc[i, j]:.1f}%"
                if status[i, j] == "stuck":
                    txt += "\n(takıldı)"
                elif status[i, j] == "exploded":
                    txt += "\n(kayıp patladı)"
                elif status[i, j] not in ("", "stable"):
                    txt += "\n" + status[i, j]
                ax.text(j, i, txt, ha="center", va="center", fontsize=8 if n_seeds > 1 else 9,
                        color=SURFACE if acc[i, j] > 55 else INK)

    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:g}" for lr in lrs])
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels(opts)
    ax.set_xlabel("Öğrenme oranı (lr)")
    ax.set_title("DeepCNN: öğrenme oranı – optimizer kararlılık haritası")

    # Hücreler arasında beyaz boşluk
    ax.set_xticks(np.arange(-0.5, len(lrs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(opts), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Son test doğruluğu (%)")
    cb.outline.set_visible(False)
    fig.text(0.5, -0.03,
             f"{n_seeds} seed; hücre değeri: sonlu koşuların ortalama son test doğruluğu.  "
             "Taralı: tüm seed'lerde NaN/Inf (kayan nokta taşması).\n'patladı': sonlu ama "
             "eğitim kaybı > 10·ln10.  'takıldı': kayıp sonlu ve küçük, doğruluk şans "
             "düzeyinde (< %15).",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    _save(fig, plots_dir, "lr_heatmap.png")


# ---------------------------------------------------------------------------
# 4) Süre vs doğruluk
# ---------------------------------------------------------------------------
def plot_time_vs_acc(runs_df, plots_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in MODEL_ORDER:
        for opt in OPT_ORDER:
            s = runs_df[(runs_df.model == model) & (runs_df.optimizer == opt)]
            s = s[np.isfinite(s.final_test_acc)]
            if s.empty:
                continue
            c = OPT_COLORS[opt]
            face = c if model == "ShallowCNN" else SURFACE  # dolu = sığ, boş = derin
            # Tek tek seed'ler (soluk)
            ax.scatter(s.total_time, s.final_test_acc * 100, s=14, color=c, alpha=0.3,
                       marker=OPT_MARKERS[opt], linewidths=0)
            # Ortalama ± std
            ax.errorbar(s.total_time.mean(), s.final_test_acc.mean() * 100,
                        xerr=np.nan_to_num(s.total_time.std()),
                        yerr=np.nan_to_num(s.final_test_acc.std() * 100),
                        fmt=OPT_MARKERS[opt], markersize=9, markerfacecolor=face,
                        markeredgecolor=c, markeredgewidth=2, ecolor=c, elinewidth=1.2,
                        capsize=3, zorder=3)

    ax.set_xlabel("Toplam eğitim süresi (s)")
    ax.set_ylabel("Son test doğruluğu (%)")
    ax.set_title("Eğitim süresi – test doğruluğu")

    opt_handles = [Line2D([0], [0], marker=OPT_MARKERS[o], color="none", markerfacecolor=OPT_COLORS[o],
                          markeredgecolor=OPT_COLORS[o], markersize=8, label=o) for o in OPT_ORDER]
    model_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor=MUTED,
               markersize=8, label="ShallowCNN (dolu)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=2, markersize=8, label="DeepCNN (boş)"),
    ]
    leg1 = ax.legend(handles=opt_handles, title="Optimizer", loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model", loc="center left")
    fig.text(0.5, -0.03, "Büyük işaretçi: seed ortalaması ± std;  küçük soluk noktalar: tek tek seed'ler",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    _save(fig, plots_dir, "time_vs_acc.png")


# ---------------------------------------------------------------------------
# 5) Sonlu fark hata analizi
# ---------------------------------------------------------------------------
DTYPE_STYLE = {
    "float64": dict(color="#2a78d6", marker="o", eps=np.finfo(np.float64).eps),
    "float32": dict(color="#eb6834", marker="s", eps=np.finfo(np.float32).eps),
}


def plot_fd_error(fd_df, plots_dir):
    models = [m for m in MODEL_ORDER if m in set(fd_df.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.3), sharey=True,
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = fd_df[fd_df.model == model]
        for dtype, st in DTYPE_STYLE.items():
            s = sub[sub.dtype == dtype].sort_values("h")
            if s.empty:
                continue
            err = np.maximum(s.rel_error.values, 1e-17)  # log eksende 0 çizilemez
            ax.plot(s.h, err, color=st["color"], marker=st["marker"], markersize=4.5, label=dtype)
            # Kuramsal en iyi adım h* ~ ε^(1/3)
            ax.axvline(st["eps"] ** (1 / 3), color=st["color"], linestyle=":", linewidth=1.2)

        # Eğim kılavuzları (float64 eğrisine sabitlenmiş): sağda ∝ h², solda ∝ 1/h
        s64 = sub[sub.dtype == "float64"].sort_values("h")
        if len(s64) > 3:
            h = s64.h.values
            e = np.maximum(s64.rel_error.values, 1e-17)
            i_r = np.argmin(np.abs(np.log10(h) + 2))      # h ≈ 1e-2
            hr = np.logspace(-4, 0, 20)
            ax.plot(hr, e[i_r] * (hr / h[i_r]) ** 2, color=MUTED, linestyle="--", linewidth=1)
            ax.text(hr[-6], e[i_r] * (hr[-6] / h[i_r]) ** 2 * 3, "∝ h²  (kesme)",
                    color=MUTED, fontsize=8.5, ha="right")
            i_l = np.argmin(np.abs(np.log10(h) + 10))     # h ≈ 1e-10
            hl = np.logspace(-12, -7, 20)
            ax.plot(hl, e[i_l] * (h[i_l] / hl), color=MUTED, linestyle="--", linewidth=1)
            ax.text(hl[8], e[i_l] * (h[i_l] / hl[8]) / 6, "∝ ε/h  (yuvarlama)",
                    color=MUTED, fontsize=8.5, ha="left")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model)
        ax.set_xlabel("Adım boyu h")
    axes[0, 0].set_ylabel("Göreli hata  |D_h − ∇L·v| / |∇L·v|")
    axes[0, 0].legend(title="Aritmetik", loc="lower left")
    fig.text(0.5, -0.03,
             "Merkezi fark D_h = [L(θ+hv) − L(θ−hv)] / 2h, referans: float64 otomatik türev.  "
             "Noktalı dikey çizgiler: kuramsal en iyi adım h* ≈ ε^(1/3).",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    _save(fig, plots_dir, "fd_error.png")


# ---------------------------------------------------------------------------
# 6) CPU / GPU
# ---------------------------------------------------------------------------
def plot_cpu_gpu(bench_df, plots_dir):
    devices = list(dict.fromkeys(bench_df.device))           # cpu, (cuda)
    names = {d: bench_df[bench_df.device == d].device_name.iloc[0] for d in devices}
    colors = {"cpu": "#2a78d6", "cuda": "#eb6834"}
    models = [m for m in MODEL_ORDER if m in set(bench_df.model)]
    x = np.arange(len(models))
    width = 0.8 / len(devices)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for k, d in enumerate(devices):
        s = bench_df[bench_df.device == d].set_index("model").loc[models]
        pos = x + (k - (len(devices) - 1) / 2) * width
        ax.bar(pos, s.ms_per_iter_mean, width=width - 0.03, yerr=s.ms_per_iter_std,
               color=colors.get(d, MUTED), label=names[d], capsize=3,
               error_kw=dict(ecolor=INK, elinewidth=1))
        for p, v, sd, row in zip(pos, s.ms_per_iter_mean, s.ms_per_iter_std, s.itertuples()):
            label = f"{v:.1f} ms"
            if d == "cuda" and "speedup_vs_cpu" in s.columns:
                label += f"\n×{row.speedup_vs_cpu:.1f}"
            # Etiket hata çubuğunun üstüne yazılır (çubukla çakışmasın)
            ax.text(p, v + np.nan_to_num(sd) + 1.5, label, ha="center", va="bottom",
                    fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("İterasyon başına süre (ms)")
    ax.set_title("CPU – GPU karşılaştırması (ileri + geri + Adam adımı)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.margins(y=0.2)
    fig.tight_layout()
    _save(fig, plots_dir, "cpu_gpu.png")


# ---------------------------------------------------------------------------
# 7) Hessian spektrumu: kuvvet yöntemi yakınsaması, λ_max ve η_kritik
# ---------------------------------------------------------------------------
def _exp2_boundary(exp2_df, opt="SGD"):
    """Deney 2'de tüm seed'lerde kararlı en büyük lr ve en az bir seed'de ıraksayan en küçük lr."""
    if exp2_df is None:
        return None, None
    g = exp2_df[exp2_df.optimizer == opt]
    ok = [lr for lr, s in g.groupby("lr") if (s.status == "stable").all()]
    bad = [lr for lr, s in g.groupby("lr") if s.status.isin(["nan_inf", "exploded"]).any()]
    return (max(ok) if ok else None), (min(bad) if bad else None)


def plot_sharpness(sharp_df, hist_df, plots_dir, exp2_df=None):
    models = [m for m in MODEL_ORDER if m in set(sharp_df.model)]
    # Kuvvet yöntemi mutlak değerce en büyük özdeğere yakınsar. Düzlük bölgesinde (ör.
    # DeepCNN'in başlangıç platosu) bu negatif bir özdeğer olabilir; o zaman λ_max
    # bilinmez ve 2/λ_max tanımsızdır. Bu ölçümler (b) ve (c)'den çıkarılır, dipnotta sayılır.
    neg = sharp_df[sharp_df.lambda_max <= 0]
    sharp_pos = sharp_df[sharp_df.lambda_max > 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) Kuvvet yönteminin yakınsaması: göreli hata |λ_k − λ_son| / |λ_son|
    ax = axes[0]
    if hist_df is not None and not hist_df.empty:
        last_ep = hist_df.epoch.max()
        for model in models:
            st = MODEL_STYLE[model]
            for ep, ls in ((0, ":"), (last_ep, st["linestyle"])):
                h = hist_df[(hist_df.model == model) & (hist_df.epoch == ep)].sort_values("k")
                if len(h) < 2:
                    continue
                lam = h.lambda_k.values
                err = np.abs(lam[:-1] - lam[-1]) / max(abs(lam[-1]), 1e-30)
                ax.plot(h.k.values[:-1], np.maximum(err, 1e-16), color=st["color"],
                        linestyle=ls, linewidth=1.6, label=f"{model}, epoch {ep}")
        ax.set_yscale("log")
        ax.set_xlabel("Kuvvet yöntemi iterasyonu k")
        ax.set_ylabel("|λ_k − λ_son| / |λ_son|")
        ax.set_title("(a) Kuvvet yönteminin yakınsaması")
        ax.legend(fontsize=8)

    # (b) λ_max'ın eğitim boyunca değişimi
    ax = axes[1]
    for model in models:
        g = sharp_pos[sharp_pos.model == model].groupby("epoch").lambda_max
        m, s = g.mean(), g.std().fillna(0)
        st = MODEL_STYLE[model]
        ax.plot(m.index, m.values, color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=5, label=model)
        lo, hi = _band(m.values, s.values)
        ax.fill_between(m.index, lo, hi, color=st["color"], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (Momentum, η = 0.01 ile eğitim)")
    ax.set_ylabel("λ_max(∇²L)")
    ax.set_title("(b) Hessian'ın en büyük özdeğeri")
    ax.legend()

    # (c) Teorik kararlılık sınırı η_kritik = 2/λ_max ve Deney 2'deki deneysel sınır
    ax = axes[2]
    for model in models:
        g = sharp_pos[sharp_pos.model == model].groupby("epoch").eta_crit_gd
        m, s = g.mean(), g.std().fillna(0)
        st = MODEL_STYLE[model]
        ax.plot(m.index, m.values, color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=5, label=f"{model}: 2/λ_max")
    ok, bad = _exp2_boundary(exp2_df, "SGD")
    if ok is not None:
        ax.axhline(ok, color=OPT_COLORS["SGD"], linestyle=":", linewidth=1.3,
                   label=f"Deney 2 (DeepCNN, SGD): kararlı η = {ok:g}")
    if bad is not None:
        ax.axhline(bad, color="#c0392b", linestyle=":", linewidth=1.3,
                   label=f"Deney 2 (DeepCNN, SGD): ıraksayan η = {bad:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("η_kritik (SGD)")
    ax.set_title("(c) Teorik kararlılık sınırı η < 2/λ_max")
    ax.legend(fontsize=8)

    note = ("λ_max: Hessian–vektör çarpımı + kuvvet yöntemi, sabit 1024 örneklik eğitim alt "
            "kümesi, seed ortalaması ± std.  Yerel ikinci derece modelde GD yalnızca "
            "η < 2/λ_max için kararlıdır.")
    if not neg.empty:
        which = ", ".join(f"{r.model} seed {r.seed} epoch {r.epoch}" for r in neg.itertuples())
        note += (f"\n(b)–(c): kuvvet yönteminin negatif baskın özdeğer verdiği {len(neg)} ölçüm "
                 f"hariç tutuldu ({which}).")
    fig.text(0.5, -0.04 if neg.empty else -0.08, note, ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    _save(fig, plots_dir, "sharpness.png")


# ---------------------------------------------------------------------------
# 8) Ablasyon: BatchNorm ve Adam yanlılık düzeltmesi
# ---------------------------------------------------------------------------
def _mean_curve(df, col="train_loss"):
    g = df.groupby("epoch")[col]
    return g.mean(), g.std().fillna(0)


def plot_ablation(exp1_epochs, exp6_epochs, plots_dir):
    panels = [
        ("(a) SGD: BatchNorm ablasyonu",
         [("DeepCNN (BN yok)", exp1_epochs, dict(model="DeepCNN", optimizer="SGD"), "#eb6834", "--"),
          ("DeepCNN + BN", exp6_epochs, dict(variant="DeepCNN+BN / SGD"), "#2a78d6", "-")]),
        ("(b) Momentum: BatchNorm ablasyonu",
         [("DeepCNN (BN yok)", exp1_epochs, dict(model="DeepCNN", optimizer="Momentum"), "#eb6834", "--"),
          ("DeepCNN + BN", exp6_epochs, dict(variant="DeepCNN+BN / Momentum"), "#2a78d6", "-")]),
        ("(c) Adam: yanlılık düzeltmesi ablasyonu",
         [("ShallowCNN, düzeltmeli", exp1_epochs, dict(model="ShallowCNN", optimizer="Adam"), "#2a78d6", "-"),
          ("ShallowCNN, düzeltmesiz", exp6_epochs, dict(variant="ShallowCNN / Adam (düzeltmesiz)"),
           "#2a78d6", ":"),
          ("DeepCNN, düzeltmeli", exp1_epochs, dict(model="DeepCNN", optimizer="Adam"), "#eb6834", "-"),
          ("DeepCNN, düzeltmesiz", exp6_epochs, dict(variant="DeepCNN / Adam (düzeltmesiz)"),
           "#eb6834", ":")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (title, lines) in zip(axes, panels):
        for label, df, filt, color, ls in lines:
            if df is None:
                continue
            sub = df
            for k, v in filt.items():
                if k not in sub.columns:
                    sub = sub.iloc[0:0]
                    break
                sub = sub[sub[k] == v]
            if sub.empty:
                continue
            m, s = _mean_curve(sub)
            ax.plot(m.index, m.values, color=color, linestyle=ls, marker="o", markersize=3.5,
                    label=label)
            lo, hi = _band(m.values, s.values)
            ax.fill_between(m.index, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Eğitim kaybı (log ölçek)")
    fig.text(0.5, -0.03, "Aynı 5 seed (aynı başlangıç ağırlıkları ve mini-batch sırası); "
             "çizgi: ortalama, gölge: ± 1 std.", ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    _save(fig, plots_dir, "ablation.png")


# ---------------------------------------------------------------------------
def _read(results_dir, name):
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        print(f"  atlandı: {path} bulunamadı")
        return None
    return pd.read_csv(path)


def main(results_dir="results", plots_dir="plots"):
    setup_style()
    os.makedirs(plots_dir, exist_ok=True)
    print("\nGrafikler üretiliyor...")

    epochs = _read(results_dir, "exp1_epochs.csv")
    layers = _read(results_dir, "exp1_layer_grads.csv")
    runs = _read(results_dir, "exp1_runs.csv")
    exp2 = _read(results_dir, "exp2_lr_stability.csv")
    fd = _read(results_dir, "exp3_fd_error.csv")
    bench = _read(results_dir, "exp4_cpu_gpu.csv")

    if fd is not None:
        plot_fd_error(fd, plots_dir)
    if bench is not None:
        plot_cpu_gpu(bench, plots_dir)
    if epochs is not None:
        plot_loss_convergence(epochs, plots_dir)
    if epochs is not None and layers is not None:
        plot_gradient_norm(layers, epochs, plots_dir)
    if exp2 is not None:
        plot_lr_heatmap(exp2, plots_dir)
    if runs is not None:
        plot_time_vs_acc(runs, plots_dir)

    sharp = _read(results_dir, "exp5_sharpness.csv")
    hist = _read(results_dir, "exp5_power_history.csv")
    if sharp is not None:
        plot_sharpness(sharp, hist, plots_dir, exp2)
    exp6 = _read(results_dir, "exp6_epochs.csv")
    if exp6 is not None:
        plot_ablation(epochs, exp6, plots_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sonuç CSV'lerinden grafik üretir")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()
    main(results_dir=a.results_dir, plots_dir=a.plots_dir)
