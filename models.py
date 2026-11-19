"""
models.py
=========
Deneylerde kullanılan iki CNN mimarisi.

* ShallowCNN : 3 konvolüsyon katmanı (sığ, kararlı referans model)
* DeepCNN    : 8 konvolüsyon katmanı, BatchNorm YOK.
               Normalizasyon katmanı bilerek kullanılmamıştır; böylece derinlik
               arttıkça geri yayılımda gradyanların katmandan katmana çarpılarak
               sönmesi (vanishing) veya patlaması (exploding) ve buna bağlı
               nümerik kararsızlıklar (NaN / Inf) gözlemlenebilir.

Girdi: 1x28x28 gri seviye görüntü (FashionMNIST), çıktı: 10 sınıf logit'i.
"""

import torch.nn as nn
import torch.nn.functional as F


class ShallowCNN(nn.Module):
    """3 konvolüsyon + 2 tam bağlantılı katmandan oluşan sığ CNN."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = F.relu(self.conv3(x))             # 7x7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DeepCNN(nn.Module):
    """
    8 konvolüsyon katmanlı, BatchNorm KULLANMAYAN derin CNN.

    Her konvolüsyon 3x3, padding=1. 2. ve 4. konvolüsyondan sonra MaxPool
    uygulanır (28 -> 14 -> 7). Ağırlıklar PyTorch'un varsayılan başlatmasıyla
    (kaiming_uniform, a=sqrt(5)) başlatılır; bu başlatma ReLU için varyansı
    korumadığından derin ağda aktivasyon ve gradyan ölçekleri katman katman
    küçülür -> gradyan sönmesi gözlemlenir. Yüksek öğrenme oranlarında ise
    ağırlık güncellemeleri büyüyüp patlamaya (NaN/Inf) yol açabilir.
    """

    # Kanal sayıları: girdi + 8 konvolüsyonun çıkışları
    CHANNELS = [1, 16, 16, 32, 32, 32, 32, 64, 64]
    POOL_AFTER = {1, 3}  # 0-indeksli: conv2 ve conv4'ten sonra havuzlama

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1)
            for cin, cout in zip(self.CHANNELS[:-1], self.CHANNELS[1:])
        )
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.CHANNELS[-1] * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i in self.POOL_AFTER:
                x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


MODEL_REGISTRY = {"ShallowCNN": ShallowCNN, "DeepCNN": DeepCNN}


def build_model(name: str) -> nn.Module:
    """İsme göre model örneği oluşturur."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model: {name}. Seçenekler: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()


def weight_layers(model: nn.Module):
    """
    Ağırlık taşıyan katmanları (Conv2d / Linear) girdiden çıktıya doğru sırayla
    döndürür. Her eleman: (okunabilir_ad, modül). Örn. ('conv1', Conv2d(...)).
    Gradyan normu ölçümlerinde 'ilk katman' = listenin başı, 'son katman' = sonu.
    """
    layers, n_conv, n_fc = [], 0, 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            n_conv += 1
            layers.append((f"conv{n_conv}", module))
        elif isinstance(module, nn.Linear):
            n_fc += 1
            layers.append((f"fc{n_fc}", module))
    return layers


if __name__ == "__main__":
    # Hızlı kontrol: parametre sayıları ve katman listesi
    import torch

    for name in MODEL_REGISTRY:
        m = build_model(name)
        out = m(torch.zeros(2, 1, 28, 28))
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name}: çıktı {tuple(out.shape)}, {n_params:,} parametre, "
              f"katmanlar = {[n for n, _ in weight_layers(m)]}")
