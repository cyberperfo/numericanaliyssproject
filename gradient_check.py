"""
gradient_check.py
=================
Geri yayılımla (otomatik türev) hesaplanan gradyanın merkezi sonlu farkla
doğrulanması ve sonlu fark yaklaşımının HATA ANALİZİ.

Kayıp fonksiyonu L: R^P -> R, birim yön vektörü v (||v||_2 = 1) için yönlü türev

    D_v L(θ) = ∇L(θ) · v                                  (otomatik türev, "kesin")

ve merkezi sonlu fark yaklaşımı

    D_h = [ L(θ + h v) − L(θ − h v) ] / (2h).

Taylor açılımından toplam hata iki bileşenlidir:

    |D_h − D_v L|  ≈  (h² / 6) |L'''|      +   ε_mach |L| / h
                      kesme hatası O(h²)        yuvarlama hatası O(ε/h)

İkisini dengeleyen en iyi adım h* ≈ (3 ε_mach |L| / |L'''|)^(1/3) ~ ε_mach^(1/3):
    float64 (ε ≈ 2.2e-16) -> h* ~ 1e-5      float32 (ε ≈ 1.2e-7) -> h* ~ 1e-2..1e-3
Grafikte log-log ölçekte sağda eğimi +2, solda eğimi −1 olan "V" şekli beklenir.

Not: ReLU ve MaxPool kırık (türevi süreksiz) noktalar içerir; büyük h bu
noktaları aşarsa kesme hatası O(h²)'den daha hızlı büyüyebilir.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model


def _set_params(params, base, direction, t):
    """θ ← θ0 + t·v  (yerinde, gradyan izlenmeden)."""
    with torch.no_grad():
        for p, p0, v in zip(params, base, direction):
            p.copy_(p0 + t * v)


def directional_fd_errors(model_name, X, y, hs, seed=42, batch=32):
    """
    Bir model için farklı h değerlerinde merkezi sonlu farkın göreli hatasını
    hem float64 hem float32 aritmetiğinde hesaplar.

    Referans (kesin) değer her zaman float64 otomatik türevden alınır.
    Dönen liste: [{model, dtype, h, fd, exact, rel_error}, ...]
    """
    torch.manual_seed(seed)
    model64 = build_model(model_name).double().cpu()
    xb64 = X[:batch].detach().cpu().double()
    yb = y[:batch].detach().cpu()

    # --- Kesin yönlü türev (float64, otomatik türev) ---
    params64 = list(model64.parameters())
    loss = F.cross_entropy(model64(xb64), yb)
    grads = torch.autograd.grad(loss, params64)
    gen = torch.Generator().manual_seed(seed + 1)
    direction = [torch.randn(p.shape, generator=gen, dtype=torch.float64) for p in params64]
    vnorm = torch.sqrt(sum((d ** 2).sum() for d in direction))
    direction = [d / vnorm for d in direction]                      # ||v||_2 = 1
    exact = float(sum((g * d).sum() for g, d in zip(grads, direction)))

    rows = []
    for dtype in (torch.float64, torch.float32):
        model = copy.deepcopy(model64).to(dtype)
        params = list(model.parameters())
        base = [p.detach().clone() for p in params]
        dirs = [d.to(dtype) for d in direction]
        xb = xb64.to(dtype)
        model.eval()
        for h in hs:
            with torch.no_grad():
                _set_params(params, base, dirs, +h)
                l_plus = F.cross_entropy(model(xb), yb).item()
                _set_params(params, base, dirs, -h)
                l_minus = F.cross_entropy(model(xb), yb).item()
            _set_params(params, base, dirs, 0.0)
            fd = (l_plus - l_minus) / (2 * h)
            rows.append(dict(model=model_name, dtype=str(dtype).replace("torch.", ""), h=h,
                             fd=fd, exact=exact,
                             rel_error=abs(fd - exact) / max(abs(exact), 1e-30)))
    return rows


def default_hs():
    """10^-12 ... 10^0 arası logaritmik h ızgarası."""
    return list(np.logspace(-12, 0, 25))
