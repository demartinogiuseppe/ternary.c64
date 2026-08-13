import torch, struct, numpy as np

CKPT, OUT = "out_FINAL_tern128/ckpt.pt", "model_ternary.bin"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd, a = ck["model"], ck["model_args"]
L, D, V = a["n_layers"], a["dim"], a["vocab_size"]
H = sd["layers.0.feed_forward.w1.weight"].shape[0]
KV = sd["layers.0.attention.wk.weight"].shape[0]

def pack(w):
    """ternarizza e impacchetta 4 pesi per byte, riga per riga"""
    s = w.abs().median().clamp(min=1e-5)
    q = (w / s).round().clamp(-1, 1).to(torch.int8).numpy() + 1   # 0,1,2
    flat = q.reshape(-1)
    assert flat.size % 4 == 0
    b = (flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6))
    return s.item(), b.astype(np.uint8)

f = open(OUT, "wb")
f.write(b"TC64")
f.write(struct.pack("<7i", 1, D, L, a["n_heads"], KV, V, H))

order = ["attention.wq", "attention.wk", "attention.wv", "attention.wo",
         "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]
nbytes = 0
for i in range(L):
    for n in order:
        s, b = pack(sd[f"layers.{i}.{n}.weight"])
        f.write(struct.pack("<f", s)); f.write(b.tobytes()); nbytes += len(b)

# RMSNorm in float32 (per ora)
for i in range(L):
    for n in ["attention_norm", "ffn_norm"]:
        f.write(sd[f"layers.{i}.{n}.weight"].float().numpy().astype("<f4").tobytes())
f.write(sd["norm.weight"].float().numpy().astype("<f4").tobytes())

# embedding / output head condivisa, int8 + scala
e = sd["tok_embeddings.weight"].float()
es = e.abs().max().item() / 127.0
f.write(struct.pack("<f", es))
f.write((e / es).round().clamp(-127, 127).to(torch.int8).numpy().tobytes())
f.close()

import os
print(f"dim={D} layers={L} heads={a['n_heads']} kv={KV} vocab={V} hidden={H}")
print(f"pesi ternari packed: {nbytes:,} byte ({nbytes/1024:.0f} KB)")
print(f"file totale: {os.path.getsize(OUT):,} byte ({os.path.getsize(OUT)/1024:.0f} KB)")