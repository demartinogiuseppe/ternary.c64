"""export_banchi.py - turns model_ternary.bin into what the cartridge needs:
one .bin per matrix ready for #embed, the transposed embedding, the scale
constants, and the EasyFlash bank allocation plan.

Run this first. Everything else reads what it writes, and gen_l04.py reads
the bank plan out of banchi.h rather than recomputing it - two allocators
that silently disagree are exactly the kind of bug this project has paid for
already.

Note: inline comments below are in Italian.

--- appunti originali -------------------------------------------------

Estrae dal modello un .bin per matrice, pronto per #embed,
e genera il piano di allocazione dei banchi EasyFlash.

Dimensioni misurate del kernel (misura_kernel.py):
    4 byte/riga  ->   226 B di codice
   32 byte/riga  ->  1620 B
   88 byte/riga  ->  4406 B

Vincoli:
  - codice e pesi condividono la finestra $8000: il kernel e le matrici che
    legge devono stare NELLO STESSO banco, entro 16384 byte
  - l'embedding e' 65536 byte: NON entra in un banco, va diviso in quattro.
    Il codice dell'output head deve quindi stare in RAM, non in un banco,
    altrimenti sparirebbe alla prima commutazione.

Produce:
  - w<layer>_<nome>.bin   una per ciascuna delle 28 matrici
  - emb0.bin .. emb3.bin  embedding/output head int8 (tied), 16384 B ciascuno
  - scale.h               scale in (mantissa, shift) + RMSNorm in Q3.12
  - banchi.h              le pragma di regione
"""

import struct
import os

CODICE = {4: 226, 32: 1620, 88: 4406}
BANCO = 16384

W = open("model_ternary.bin", "rb").read()
assert W[:4] == b"TC64"
ver, D, L, NH, KV, V, H = struct.unpack_from("<7i", W, 4)
print(f"dim={D} layers={L} heads={NH} kvdim={KV} vocab={V} hidden={H}\n")

FORME = [("wq", D, D), ("wk", KV, D), ("wv", KV, D), ("wo", D, D),
         ("w1", H, D), ("w2", D, H), ("w3", H, D)]

off = 4 + 7 * 4
matrici = []

for l in range(L):
    for nome, righe, col in FORME:
        scala = struct.unpack_from("<f", W, off)[0]
        off += 4
        n = righe * col // 4
        open(f"w{l}_{nome}.bin", "wb").write(W[off:off + n])
        off += n
        matrici.append((f"L{l}_{nome}", righe, col, col // 4, n, scala,
                        f"w{l}_{nome}.bin"))

norm = []
for l in range(L):
    for nome in ("an", "fn"):
        norm.append((f"L{l}_{nome}", struct.unpack_from(f"<{D}f", W, off)))
        off += 4 * D
norm.append(("final", struct.unpack_from(f"<{D}f", W, off)))
off += 4 * D

emb_scala = struct.unpack_from("<f", W, off)[0]
off += 4
emb = W[off:off + V * D]
off += V * D
assert off == len(W), f"disallineato: {off} vs {len(W)}"
print(f"file letto per intero: {off} byte")

# l'embedding non entra in un banco: quattro pezzi da 16384
NEMB = (len(emb) + BANCO - 1) // BANCO
for i in range(NEMB):
    open(f"emb{i}.bin", "wb").write(emb[i * BANCO:(i + 1) * BANCO])
print(f"embedding {len(emb)} B -> {NEMB} pezzi da {BANCO} B "
      f"({BANCO // D} token ciascuno)")

# ---- embedding TRASPOSTO, per l'output head ----
# Il MAC 16x8 (sec. 9-ter) vuole l'operando a 16 bit fisso nell'anello
# interno.  Nell'output head il vettore fisso e' xb, quindi l'anello interno
# deve scorrere i TOKEN a dimensione fissa: serve emb[d][token], non
# emb[token][d].
# Una riga trasposta e' V byte, e BANCO/V righe stanno esattamente in un
# banco: nessuna riga a cavallo di due banchi.  Se il conto non tornasse,
# l'output head dovrebbe commutare banco a meta' riga.
assert BANCO % V == 0, f"una riga trasposta da {V} B non divide il banco"
# ⚠️ memorizzato SPOSTATO di +128, cioe' xor 0x80 su un byte in complemento
# a due: cosi' l'operando a 8 bit e' senza segno e la tabella dei quarti di
# quadrato basta com'e' (sec. 9-ter).  La correzione -128*somma(xb) si
# applica una volta per token, non per elemento.
embT = bytearray(len(emb))
for t in range(V):
    riga = emb[t * D:(t + 1) * D]
    for d in range(D):
        embT[d * V + t] = riga[d] ^ 0x80
NEMBT = len(embT) // BANCO
for i in range(NEMBT):
    open(f"embT{i}.bin", "wb").write(bytes(embT[i * BANCO:(i + 1) * BANCO]))
print(f"embedding trasposto -> {NEMBT} pezzi da {BANCO} B "
      f"({BANCO // V} dimensioni ciascuno, righe da {V} B)\n")


def mkscale(s):
    e = 0
    while s < 0.5 and e < 40:
        s *= 2.0
        e += 1
    return int(round(s * 32768.0)), 15 + e


# ---- allocazione: primo adattamento ----
banchi = []

def spazio(b, bpr, dim):
    extra = 0 if bpr in b[1] else CODICE[bpr]
    return b[0] + extra + dim

for nome, righe, col, bpr, dim, scala, fn in matrici:
    for b in banchi:
        if spazio(b, bpr, dim) <= BANCO:
            if bpr not in b[1]:
                b[0] += CODICE[bpr]
                b[1].add(bpr)
            b[0] += dim
            b[2].append((nome, dim, bpr, fn))
            break
    else:
        banchi.append([CODICE[bpr] + dim, {bpr}, [(nome, dim, bpr, fn)]])

NB = len(banchi)
print(f"{'banco':>6} {'kernel':>8} {'usato':>7} {'libero':>7}  matrici")
print("-" * 78)
for i, b in enumerate(banchi, start=1):
    ks = "+".join(str(k) for k in sorted(b[1]))
    print(f"{i:>6} {ks:>8} {b[0]:>6}B {BANCO - b[0]:>6}B  "
          + ", ".join(n for n, _, _, _ in b[2]))
for i in range(NEMB):
    print(f"{NB + 1 + i:>6} {'-':>8} {BANCO:>6}B {0:>6}B  "
          f"embedding parte {i} (token {i * BANCO // D}-"
          f"{(i + 1) * BANCO // D - 1})")

tot = NB + NEMB
libero = sum(BANCO - b[0] for b in banchi)
print(f"\n{tot} banchi su 64. Spreco nei banchi delle matrici: {libero} B "
      f"({libero / 1024:.0f} KB) — irrilevante, restano {64 - tot} banchi liberi.")

# ---- scale.h ----
with open("scale.h", "w") as g:
    g.write("/* generato da export_banchi.py */\n")
    g.write(f"#define NMAT {len(matrici)}\n")
    g.write(f"#define NEMB {NEMB}\n")
    g.write(f"#define EMB_PER_BANCO {BANCO // D}   /* token per banco */\n\n")
    g.write("const unsigned MANT[NMAT] = {")
    g.write(",".join(str(mkscale(s)[0]) for *_, s, _ in matrici))
    g.write("};\n")
    g.write("const unsigned char SHIFT[NMAT] = {")
    g.write(",".join(str(mkscale(s)[1]) for *_, s, _ in matrici))
    g.write("};\n\n")
    m, sh = mkscale(emb_scala)
    g.write(f"#define EMB_MANT {m}\n#define EMB_SHIFT {sh}\n\n")
    g.write("/* RMSNorm in Q3.12, restano in RAM: 9 x 128 interi = 2304 B */\n")
    for nome, vals in norm:
        q = [int(round(v * 4096.0)) for v in vals]
        g.write(f"const int NRM_{nome}[{D}] = {{" + ",".join(map(str, q)) + "};\n")

# ---- banchi.h ----
with open("banchi.h", "w") as g:
    g.write("/* generato da export_banchi.py */\n\n")
    for i, b in enumerate(banchi, start=1):
        g.write(f"#pragma section( bcode{i}, 0 )\n")
        g.write(f"#pragma section( bdata{i}, 0 )\n")
        g.write(f"#pragma region( bank{i}, 0x8000, 0xc000, , {i}, "
                f"{{ bcode{i}, bdata{i} }} )\n")
    g.write("\n/* embedding: solo dati, il codice dell'output head sta in RAM */\n")
    for i in range(NEMB):
        n = NB + 1 + i
        g.write(f"#pragma section( bdataE{i}, 0 )\n")
        g.write(f"#pragma region( bankE{i}, 0x8000, 0xc000, , {n}, "
                f"{{ bdataE{i} }} )\n")
    g.write("\n/* contenuto:\n")
    for i, b in enumerate(banchi, start=1):
        for nome, dim, bpr, fn in b[2]:
            g.write(f"   banco {i:2d}: {nome:10s} {dim:6d} B  kernel {bpr:2d}  {fn}\n")
    for i in range(NEMB):
        g.write(f"   banco {NB + 1 + i:2d}: embedding parte {i}   {BANCO} B  emb{i}.bin\n")
    g.write("*/\n")

print("scritti: 28 .bin delle matrici, "
      f"{NEMB} .bin dell'embedding, scale.h, banchi.h")