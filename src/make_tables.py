import numpy as np

def report(nome, ref, appr, unit=""):
    e = np.abs(appr - ref)
    rel = e.max() / (np.abs(ref).max() + 1e-12)
    print(f"{nome:28s} errore max {e.max():.6f}{unit}   relativo {rel:.3%}")

# ---- SiLU: 256 voci su [-8, 8], passo 1/16 ----
idx = np.arange(256)
xs = (idx - 128) / 24.0
silu = xs / (1.0 + np.exp(-xs))
tab_silu = np.clip(np.round(silu * 1024), -32768, 32767).astype(np.int16)

xt = np.linspace(-3.1, 4.3, 20000)                 # range reale osservato
ref = xt / (1.0 + np.exp(-xt))
i = np.clip(np.round(xt * 24.0) + 128, 0, 255).astype(int)
report("SiLU (tabella 256)", ref, tab_silu[i] / 1024.0)

# ---- exp per softmax: 512 voci su [0, 8], passo 1/64 ----
d = np.arange(512) / 64.0
tab_exp = np.clip(np.round(np.exp(-d) * 32767), 0, 32767).astype(np.int16)
dt = np.linspace(0, 8, 20000)
j = np.clip(np.round(dt * 64), 0, 511).astype(int)
report("exp(-d) (tabella 512)", np.exp(-dt), tab_exp[j] / 32767.0)

# ---- 1/sqrt: normalizzazione + 256 voci sulla mantissa in [1, 4) ----
m = 1.0 + 3.0 * np.arange(256) / 256.0
tab_rsq = np.round(1.0 / np.sqrt(m) * 16384).astype(np.int16)               # Q1.14

def rsqrt_fx(v):
    sh = 0
    while v < 1.0:  v *= 4.0; sh += 1
    while v >= 4.0: v /= 4.0; sh -= 1
    k = min(int((v - 1.0) / 3.0 * 256), 255)
    return (tab_rsq[k] / 16384.0) * (2.0 ** sh)

vt = np.exp(np.linspace(np.log(0.005), np.log(2.5), 4000))
ap = np.array([rsqrt_fx(v) for v in vt])
report("1/sqrt (tabella 256)", 1.0 / np.sqrt(vt), ap)

# ---- cos/sin RoPE in Q1.14 ----
#
# ⚠️ QUI C'ERA IL BUG.  rope.bin contiene 1024 float di COSENO seguiti da 1024
# di SENO (128 posizioni x 8 frequenze, per ciascuno).  La riga era
#     cs = r[:1024]
# cioe' prendeva i soli coseni e buttava via tutti i seni.  tables.bin usciva
# senza seno, generate_fx.c leggeva T_cos e poi trovava la fine del file, e
# T_sin restava tutto a ZERO.  La rotazione RoPE diventava una scalatura per il
# coseno.  Il testo divergeva dal riferimento a posizione 12: divergenza
# attribuita per errore alla quantizzazione.  Rimessi i seni, la sequenza dei
# 35 token e' IDENTICA al riferimento float.
#
# SEQ e' il numero di posizioni da tabellare = max_seq_len effettivo del C64.
# Costa SEQ * HALF * 2 tabelle * 2 byte.  Con SEQ=64 sono 2048 byte, e il
# totale di tables.bin resta nei 4096 byte previsti dal piano di memoria.

SEQ_FILE = 128        # posizioni presenti in rope.bin
HALF     = 8          # coppie ruotate per testa (head_dim 16 / 2)
SEQ      = 64         # posizioni da mettere in tables.bin

r = np.fromfile("rope.bin", dtype="<f4")
assert len(r) == 2 * SEQ_FILE * HALF, f"rope.bin inatteso: {len(r)} float"
cos_t = r[:SEQ_FILE * HALF].reshape(SEQ_FILE, HALF)[:SEQ]
sin_t = r[SEQ_FILE * HALF:].reshape(SEQ_FILE, HALF)[:SEQ]

report("cos RoPE (Q1.14)", cos_t, np.round(cos_t * 16384) / 16384.0)
report("sin RoPE (Q1.14)", sin_t, np.round(sin_t * 16384) / 16384.0)

np.savez("tables.npz", silu=tab_silu, exp=tab_exp, rsqrt=tab_rsq)

with open("tables.bin", "wb") as g:
    g.write(tab_silu.astype("<i2").tobytes())              # 256 x int16, Q5.10
    g.write(tab_exp.astype("<i2").tobytes())               # 512 x int16, Q0.15
    g.write(tab_rsq.astype("<i2").tobytes())               # 256 x int16, Q1.14
    g.write(np.round(cos_t * 16384).astype("<i2").tobytes())   # SEQ*HALF, Q1.14
    g.write(np.round(sin_t * 16384).astype("<i2").tobytes())   # SEQ*HALF, Q1.14

import os
print(f"\nSEQ = {SEQ} posizioni  ->  RoPE {2 * SEQ * HALF * 2} byte")
print("tables.bin:", os.path.getsize("tables.bin"), "byte")
print("ordine: silu 256 | exp 512 | 1/sqrt 256 | cos SEQ*8 | sin SEQ*8")