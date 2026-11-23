"""
optimizers.py
=============
SGD, Momentum, RMSprop ve Adam optimizasyon algoritmalarının formüllerinden
SIFIRDAN yazılmış uygulamaları. Her sınıf torch.optim.Optimizer arayüzünü
kullanır; böylece eğitim döngüsü değişmeden kullanılabilir.

Gösterim:  θ_t  parametre vektörü,  g_t = ∇_θ L(θ_{t-1})  mini-batch gradyanı,
           η  öğrenme oranı,  ⊙  eleman bazlı çarpım,  t = 1, 2, ...

    SGD        θ_t = θ_{t-1} − η g_t

    Momentum   v_t = μ v_{t-1} + g_t                       (v_0 = 0, μ = 0.9)
               θ_t = θ_{t-1} − η v_t

    RMSprop    s_t = α s_{t-1} + (1 − α) g_t ⊙ g_t         (s_0 = 0, α = 0.99)
               θ_t = θ_{t-1} − η g_t / (√s_t + ε)

    Adam       m_t = β1 m_{t-1} + (1 − β1) g_t             (β1 = 0.9)
               v_t = β2 v_{t-1} + (1 − β2) g_t ⊙ g_t       (β2 = 0.999)
               m̂_t = m_t / (1 − β1^t),   v̂_t = v_t / (1 − β2^t)   (yanlılık düzeltmesi)
               θ_t = θ_{t-1} − η m̂_t / (√v̂_t + ε)

Hesap karmaşıklığı: P parametre sayısı olmak üzere her adım O(P) zaman alır.
Ek bellek: SGD O(1), Momentum O(P), RMSprop O(P), Adam O(2P).

İşlem sırası PyTorch'un referans uygulamasıyla aynı tutulmuştur; tests.py
içindeki birim testleri iki uygulamanın kayan nokta hassasiyetinde
eşleştiğini doğrular.
"""

import math

import torch
from torch.optim import Optimizer


class ManualSGD(Optimizer):
    """Saf SGD ve (momentum > 0 ise) ağır top (heavy-ball) momentumlu SGD."""

    def __init__(self, params, lr=0.01, momentum=0.0):
        if lr <= 0 or momentum < 0:
            raise ValueError("lr > 0 ve momentum >= 0 olmalı")
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if mu > 0:
                    state = self.state[p]
                    if "v" not in state:
                        state["v"] = g.clone()            # v_1 = g_1  (v_0 = 0)
                    else:
                        state["v"].mul_(mu).add_(g)       # v_t = μ v_{t-1} + g_t
                    g = state["v"]
                p.add_(g, alpha=-lr)                      # θ_t = θ_{t-1} − η (·)
        return None


class ManualRMSprop(Optimizer):
    """RMSprop: gradyan karelerinin üstel hareketli ortalaması ile ölçekleme."""

    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8):
        if lr <= 0 or not 0 <= alpha < 1:
            raise ValueError("lr > 0 ve 0 <= alpha < 1 olmalı")
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, a, eps = group["lr"], group["alpha"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "s" not in state:
                    state["s"] = torch.zeros_like(p)
                s = state["s"]
                s.mul_(a).addcmul_(g, g, value=1 - a)     # s_t = α s + (1−α) g²
                denom = s.sqrt().add_(eps)                # √s_t + ε
                p.addcdiv_(g, denom, value=-lr)           # θ −= η g / denom
        return None


class ManualAdam(Optimizer):
    """
    Adam: birinci ve ikinci moment tahminleri + yanlılık düzeltmesi.

    bias_correction=False (ABLASYON): m̂ = m, v̂ = v alınır. m_0 = v_0 = 0 başlangıcı
    nedeniyle E[m_t] = (1 − β1^t) E[g] ve E[v_t] = (1 − β2^t) E[g²] olduğundan, düzeltme
    yapılmazsa ilk adımın büyüklüğü η(1 − β1)/√(1 − β2) ≈ 3.16 η olur (β1=0.9, β2=0.999);
    yani ilk iterasyonlarda etkin öğrenme oranı ~3 kat büyür.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, bias_correction=True):
        if lr <= 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("lr > 0 ve 0 <= beta < 1 olmalı")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      bias_correction=bias_correction))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]

                m.lerp_(g, 1 - b1)                        # m_t = β1 m + (1−β1) g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)   # v_t = β2 v + (1−β2) g²

                if group["bias_correction"]:
                    bc1 = 1 - b1 ** t                     # yanlılık düzeltme katsayıları
                    bc2 = 1 - b2 ** t
                else:
                    bc1 = bc2 = 1.0                       # ablasyon: düzeltme yok
                # η m̂/(√v̂ + ε) = (η/bc1) · m / (√v/√bc2 + ε)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return None


def make_optimizer(name: str, params, lr: float, impl: str = "manual") -> Optimizer:
    """
    name : 'SGD' | 'Momentum' | 'RMSprop' | 'Adam'
    impl : 'manual' -> bu dosyadaki sıfırdan uygulamalar
           'torch'  -> PyTorch referans uygulamaları (doğrulama için)
    """
    if impl == "manual":
        table = {
            "SGD": lambda: ManualSGD(params, lr=lr),
            "Momentum": lambda: ManualSGD(params, lr=lr, momentum=0.9),
            "RMSprop": lambda: ManualRMSprop(params, lr=lr),
            "Adam": lambda: ManualAdam(params, lr=lr),
            "Adam-noBC": lambda: ManualAdam(params, lr=lr, bias_correction=False),
        }
    elif impl == "torch":
        table = {
            "SGD": lambda: torch.optim.SGD(params, lr=lr),
            "Momentum": lambda: torch.optim.SGD(params, lr=lr, momentum=0.9),
            "RMSprop": lambda: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-8),
            "Adam": lambda: torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8),
        }
    else:
        raise ValueError(f"Bilinmeyen uygulama: {impl}")
    if name not in table:
        raise ValueError(f"Bilinmeyen optimizer: {name}")
    return table[name]()
