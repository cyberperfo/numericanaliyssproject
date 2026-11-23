"""
hessian.py
==========
Kayıp fonksiyonunun Hessian matrisinin en büyük özdeğeri λ_max'ın KUVVET YÖNTEMİ
(power iteration) ile kestirimi ve bundan öğrenme oranı için kararlılık sınırı.

1) Hessian–vektör çarpımı (Pearlmutter, 1994)
   H = ∇²L(θ) ∈ R^{P×P} açıkça oluşturulmaz (P ≈ 10^5 için P² ≈ 10^10 eleman).
   Bunun yerine Hv, gradyanın v yönündeki türevi olarak iki geri yayılımla hesaplanır:
       H v = ∇_θ ( ∇_θ L(θ) · v )
   Maliyet: yaklaşık 2 gradyan hesabı, bellek O(P).

2) Kuvvet yöntemi
       v_0 rastgele, ||v_0||_2 = 1
       w_k = H v_{k-1}
       λ_k = v_{k-1}ᵀ w_k            (Rayleigh bölümü)
       v_k = w_k / ||w_k||_2
   H simetrik ve |λ_1| > |λ_2| ise Rayleigh bölümünün hatası
       |λ_k − λ_1| = O( |λ_2 / λ_1|^{2k} ).
   Durdurma ölçütü: |λ_k − λ_{k-1}| ≤ tol · |λ_k|.
   Toplam maliyet: K iterasyon × O(N·P)  (N = örnek sayısı, P = parametre sayısı).

3) Kararlılık sınırı (yerel ikinci derece model L(θ) ≈ L* + ½ (θ−θ*)ᵀ H (θ−θ*))
   Gradyan inişi:  e_{k+1} = (I − ηH) e_k  →  özdeğer yönünde çarpan (1 − ηλ_i).
       |1 − ηλ_i| < 1  ∀i   ⇔   0 < η < 2 / λ_max
   Ağır top momentumu (μ):  0 < η < 2(1 + μ) / λ_max.
   Bu sınırların ötesinde en dik eğrilik yönünde hata her adımda büyür (ıraksama).
"""

import torch
import torch.nn.functional as F


def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def hvp(loss_fn, params, vec):
    """Hessian–vektör çarpımı Hv (iki geri yayılım, Hessian oluşturulmaz)."""
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return torch.autograd.grad(_dot(grads, vec), params)


def top_eigenvalue(loss_fn, params, n_iter=100, tol=1e-3, seed=0):
    """
    Kuvvet yöntemiyle Hessian'ın en büyük (mutlak değerce) özdeğeri.

    loss_fn : parametre almayan, skaler kayıp döndüren fonksiyon (closure)
    Dönen  : (λ, [λ_1, λ_2, ...] iterasyon geçmişi)
    """
    params = [p for p in params if p.requires_grad]
    gen = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=gen).to(device=p.device, dtype=p.dtype) for p in params]
    norm = torch.sqrt(_dot(v, v))
    v = [x / norm for x in v]

    history = []
    lam = 0.0
    for _ in range(n_iter):
        w = hvp(loss_fn, params, v)
        lam = float(_dot(v, w))                       # Rayleigh bölümü vᵀHv
        history.append(lam)
        norm = float(torch.sqrt(_dot(w, w)))
        if norm == 0.0:                               # H v = 0 (ör. ölü ağ)
            break
        v = [x.detach() / norm for x in w]
        if len(history) > 1 and abs(history[-1] - history[-2]) <= tol * abs(history[-1]):
            break
    return lam, history


def model_sharpness(model, X, y, n_iter=100, tol=1e-3, seed=0):
    """Çapraz entropi kaybının (X, y üzerinde) Hessian'ının λ_max'ı."""
    was_training = model.training
    model.eval()
    lam, hist = top_eigenvalue(lambda: F.cross_entropy(model(X), y),
                               list(model.parameters()), n_iter=n_iter, tol=tol, seed=seed)
    model.train(was_training)
    return lam, hist


def critical_lr(lam, momentum=0.0):
    """Yerel ikinci derece modele göre kararlılık sınırı: η_kritik = 2(1+μ)/λ_max."""
    return 2.0 * (1.0 + momentum) / lam if lam > 0 else float("inf")
