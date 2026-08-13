import os
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """Pesi ternari {-1, 0, +1}, scala basata sulla mediana, gradiente via STE."""

    def forward(self, x):
        w = self.weight
        s = w.abs().median().clamp(min=1e-5)
        wq = (w / s).round().clamp(-1, 1) * s
        wq = w + (wq - w).detach()
        return F.linear(x, wq, self.bias)


def apply(model):
    if os.environ.get("TERNARY", "0") != "1":
        print(">>> BITPATCH: TERNARY=0 -> baseline in piena precisione")
        return model
    n = 0
    for layer in model.layers:
        for mod in (layer.attention, layer.feed_forward):
            for _, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    child.__class__ = BitLinear
                    n += 1
    print(f">>> BITPATCH: TERNARY=1 -> {n} strati ternarizzati")
    return model