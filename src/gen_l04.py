"""gen_l04.py - writes l04.c: THE TEST BENCH. All four layers plus the
output head, with every stage compared against the oracle.

This is the file that answers "is it still correct". It runs 16 consecutive
positions and checks 21 stages at each one - the output of every matrix, the
attention, the logits and the argmax - against targets read straight from
oracolo_att.exe, so the program and the reference cannot drift apart.

    336 comparisons, 0 errors.

gen_testo.py imports this module for the parts they share: the bank plan,
the ternary kernel, the 16x8 MAC and the scales. Change something here and
gen.c changes with it, which is the point.

⚠️ matvec, attenzione and rmsnorm are DUPLICATED in gen_testo.py. Touch one
   and touch the other; the proof that they agree is that l04 gives 336/0
   and gen writes the sentence.

Note: inline comments below are in Italian.

--- appunti originali -------------------------------------------------

E' il punto 3 della lista di ripresa, l'ultimo rischio strutturale prima
dell'output head.  Quello che cambia rispetto a l0kern.c (un solo layer) non
e' il numero di layer: e' che tutto diventa indicizzato.

  - 28 matrici invece di 7, sparse su 15 banchi secondo il piano gia'
    generato e verificato da export_banchi.py.  Il piano si LEGGE da
    banchi.h, non si ricalcola qui: due allocatori che divergono in silenzio
    sono esattamente il genere di bug che questo progetto ha gia' pagato.
  - un kernel srotolato PER BANCO, insieme ai pesi che legge (sec. 12).
    Nel piano nessun banco mescola larghezze diverse, quindi il banco
    identifica il kernel: 11 banchi a 32 byte/riga, 4 a 88.
  - cache KV per layer: signed char KC[LAYERS][NPOS][KVDIM] (verifica XV).
  - LE COSTANTI IN UN BANCO, non in RAM (sec. 14).  Tabelle di lookup, RoPE,
    pesi RMSNorm, righe di embedding e bersagli: 8-9 KB che in RAM non ci
    starebbero.  Vanno tutte nel banco 20.

⚠️ REGOLA DEL BANCO, l'unica cosa da tenere a mente qui: ogni funzione che
   legge una costante commuta su BANCOK come prima istruzione; matvec
   commuta sul banco dei suoi pesi.  Nessuno "ripristina" niente, perche'
   chi legge commuta sempre da solo.  Codice e dati condividono la finestra
   $8000 e il codice di tutto questo sta in RAM, quindi puo' leggere
   qualunque banco - tranne il kernel, che vive nel banco dei suoi pesi.

Bersagli: oracolo_att.exe -dump4, quattro stadi per layer (xb, q, cat, x2)
piu' x0, cioe' 17 stadi per posizione come le verifiche precedenti.
⚠️ Ricompilare l'oracolo dopo averlo toccato: gcc -O2 -o oracolo_att.exe
   oracolo_att.c -lm.  Altrimenti si generano bersagli vecchi.

Uso:  python gen_l04.py
      .\\c64.ps1 l04 -Cicli 250000000
"""

import os
import re
import struct
import subprocess
import sys

D = 128
LAYERS = 4
KVDIM = 16
HIDDEN = 352
HALF = 8
SEQ = 64
NPOS = 16          # posizioni verificate; 16 righe stanno nello schermo
BANCOK = 20        # banco delle costanti: tabelle, norm, embedding, bersagli
BANCOM = 21        # banco del MAC 16x8 e delle sue quattro tabelle (sec. 9-ter)
BANCOE = 22        # primo dei quattro banchi dell'embedding TRASPOSTO
VOCAB = 512
BANCO = 16384
DPB = BANCO // VOCAB          # dimensioni per banco: 32 righe da 512 B
NT2 = LAYERS * NPOS   # celle di cache: l'indice e' l*NPOS + t, sta in un byte

# ordine delle matrici dentro un layer, lo stesso di export_banchi.py:
# e' l'ordine con cui sono indicizzate MANT[] e SHIFT[]
ORDINE = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]

STADI = (["x0"]
         + ["L%d%s" % (l, s) for l in range(LAYERS)
            for s in ("xb", "q", "cat", "x2")]
         # output head: norm finale, il logit del token 0 (bersaglio stretto,
         # localizza l'errore su un solo prodotto scalare), somma, argmax
         + ["xbf", "lg0", "lg", "am"])

# --------------------------------------------------- piano dei banchi
# Si legge da banchi.h, prodotto da export_banchi.py.  Non ricalcolarlo.
if not os.path.exists("banchi.h"):
    print("manca banchi.h: esegui prima  python export_banchi.py")
    sys.exit(1)

testo = open("banchi.h", encoding="utf-8").read()
righe = re.findall(
    r"banco\s+(\d+):\s+L(\d)_(\w+)\s+(\d+) B\s+kernel\s+(\d+)\s+(\S+)", testo)
if not righe:
    print("banchi.h non contiene il piano: rigenerare con export_banchi.py")
    sys.exit(1)

# indice l*7+k -> (banco, byte, bpr, file)
piano = {}
banco_bpr = {}
for banco, l, nome, nbyte, bpr, fn in righe:
    banco, l, nbyte, bpr = int(banco), int(l), int(nbyte), int(bpr)
    idx = l * 7 + ORDINE.index(nome)
    piano[idx] = (banco, nbyte, bpr, fn)
    if banco in banco_bpr:
        assert banco_bpr[banco] == bpr, (
            "il banco %d mescola larghezze %d e %d: il banco non basta piu' a "
            "identificare il kernel" % (banco, banco_bpr[banco], bpr))
    else:
        banco_bpr[banco] = bpr

assert len(piano) == LAYERS * 7, "banchi.h descrive %d matrici su %d" % (
    len(piano), LAYERS * 7)
BANCHI = sorted(banco_bpr)
assert BANCOK not in BANCHI, "il banco %d e' gia' occupato dai pesi" % BANCOK
print("piano letto da banchi.h: %d matrici su %d banchi (%s)"
      % (len(piano), len(BANCHI),
         ", ".join("%d:%db" % (b, banco_bpr[b]) for b in BANCHI)))

# ------------------------------------------------- bersagli dall'oracolo
try:
    uscita = subprocess.run(["./oracolo_att.exe", "-dump4"],
                            capture_output=True, text=True, check=True).stdout
except (OSError, subprocess.CalledProcessError) as e:
    print("non riesco a eseguire oracolo_att.exe -dump4:", e)
    print("compilalo con:  gcc -O2 -o oracolo_att.exe oracolo_att.c -lm")
    sys.exit(1)

blocchi = re.split(r"--- pos (\d+), token (\d+) ---", uscita)
bersagli, token = {}, {}
for i in range(1, len(blocchi) - 2, 3):
    pos = int(blocchi[i])
    token[pos] = int(blocchi[i + 1])
    d = {}
    for riga in blocchi[i + 2].splitlines():
        m = re.match(r"\s+(\w+)\s+(-?\d+)", riga)
        if m:
            d[m.group(1)] = int(m.group(2))
    bersagli[pos] = d

for p in range(NPOS):
    assert p in bersagli, "l'oracolo non ha stampato la posizione %d" % p
    mancanti = [n for n in STADI if n not in bersagli[p]]
    assert not mancanti, "posizione %d, bersagli mancanti: %s" % (p, mancanti)

print("bersagli letti: %d posizioni x %d stadi, token %s"
      % (NPOS, len(STADI), [token[p] for p in range(NPOS)]))
print("  x2 dell'ultimo layer: %s"
      % [bersagli[p]["L3x2"] for p in range(NPOS)])

# ------------------------------------------------------------- embedding
mod = open("model_ternary.bin", "rb").read()
emb = mod[len(mod) - 512 * D:]


def riga_emb(tok):
    r = emb[tok * D:(tok + 1) * D]
    return [b - 256 if b > 127 else b for b in r]


# --------------------------------------------------------------- tabelle
t = open("tables.bin", "rb").read()
tsilu = struct.unpack_from("<256h", t, 0)
texp = struct.unpack_from("<512h", t, 256 * 2)
trsq = struct.unpack_from("<256h", t, (256 + 512) * 2)
base_cos = (256 + 512 + 256) * 2
base_sin = base_cos + SEQ * HALF * 2
cos = [struct.unpack_from("<%dh" % HALF, t, base_cos + p * HALF * 2)
       for p in range(NPOS)]
sin = [struct.unpack_from("<%dh" % HALF, t, base_sin + p * HALF * 2)
       for p in range(NPOS)]

# -------------------------------------------------- scale.h: MANT, SHIFT, NRM
if not os.path.exists("scale.h"):
    print("manca scale.h: esegui prima  python export_banchi.py")
    sys.exit(1)
sc = open("scale.h", encoding="utf-8").read()


def lista(nome):
    m = re.search(r"\b%s\[[^\]]*\]\s*=\s*\{([^}]*)\}" % nome, sc)
    assert m, "scale.h non contiene %s" % nome
    return [int(v) for v in m.group(1).split(",")]


MANT = lista("MANT")
SHIFT = lista("SHIFT")
assert len(MANT) == LAYERS * 7 and len(SHIFT) == LAYERS * 7
NRM_AN = [lista("NRM_L%d_an" % l) for l in range(LAYERS)]
NRM_FN = [lista("NRM_L%d_fn" % l) for l in range(LAYERS)]
NRM_FINAL = lista("NRM_final")

# somma_d (emb[t][d] + 128) per ogni token: e' la correzione dello
# spostamento dell'embedding trasposto, e dipende solo dal modello.
ESUM = [sum((emb[t * D + d] ^ 0x80) for d in range(D)) for t in range(VOCAB)]
assert max(ESUM) < 65536
EMB_MANT = int(re.search(r"#define EMB_MANT (\d+)", sc).group(1))
EMB_SHIFT = int(re.search(r"#define EMB_SHIFT (\d+)", sc).group(1))
print("scale.h: %d mantisse, EMB_MANT %d shift %d"
      % (len(MANT), EMB_MANT, EMB_SHIFT))


def vet(v):
    return ",".join(str(x) for x in v)


# ------------------------------------------------------------- l04data.h
with open("l04data.h", "w", encoding="utf-8") as g:
    g.write("/* generato da gen_l04.py - non modificare a mano */\n")
    g.write("#define NPOS %d\n" % NPOS)
    g.write("#define NSTADI %d\n" % len(STADI))
    g.write("#define BANCOK %d\n" % BANCOK)
    g.write("#define BANCOM %d\n" % BANCOM)
    g.write("#define NT2 %d   /* celle di cache: l*NPOS + t, sta in un byte */\n"
            % NT2)
    g.write("#define VOC %d\n" % VOCAB)
    g.write("#define BANCOE %d\n" % BANCOE)
    g.write("#define DPB %d    /* dimensioni per banco di embedding */\n" % DPB)
    g.write("#define EMB_MANT %d\n" % EMB_MANT)
    g.write("#define EMB_SHIFT %d\n\n" % EMB_SHIFT)
    g.write("/* stadi, in ordine: %s */\n\n" % " ".join(STADI))

    g.write("/* --- in RAM: 84 byte, si leggono a ogni riga --- */\n")
    g.write("const unsigned MANTM[%d] = {%s};\n" % (len(MANT), vet(MANT)))
    g.write("const unsigned char SHIFTM[%d] = {%s};\n\n"
            % (len(SHIFT), vet(SHIFT)))

print("l04data.h scritto")

# l04cost.h: le costanti, incluse DENTRO la sezione dati del banco.  Un file
# a parte e non una macro: qui ci sono ~9 KB di numeri.
with open("l04cost.h", "w", encoding="utf-8") as g:
    g.write("/* generato da gen_l04.py - va incluso dentro "
            "#pragma data( bdatak ) */\n")
    g.write("const int SILU[256] = {%s};\n" % vet(tsilu))
    g.write("const int TEXP[512] = {%s};\n" % vet(texp))
    g.write("const int RSQ[256] = {%s};\n" % vet(trsq))
    g.write("const int RCOS[%d][8] = {%s};\n"
            % (NPOS, ",".join("{%s}" % vet(c) for c in cos)))
    g.write("const int RSIN[%d][8] = {%s};\n"
            % (NPOS, ",".join("{%s}" % vet(s) for s in sin)))
    g.write("const int NRM_AN[%d][%d] = {%s};\n"
            % (LAYERS, D, ",".join("{%s}" % vet(v) for v in NRM_AN)))
    g.write("const int NRM_FN[%d][%d] = {%s};\n"
            % (LAYERS, D, ",".join("{%s}" % vet(v) for v in NRM_FN)))
    g.write("const int NRM_FIN[%d] = {%s};\n" % (D, vet(NRM_FINAL)))
    # somma_d eu[d][t] per ogni token: la correzione dello spostamento +128
    # dell'embedding.  Dipende solo dal modello, quindi si calcola qui.
    g.write("const unsigned ESUM[%d] = {%s};\n" % (VOCAB, vet(ESUM)))
    g.write("const int EMBT[%d][%d] = {%s};\n"
            % (NPOS, D, ",".join("{%s}" % vet(riga_emb(token[p]))
                                 for p in range(NPOS))))
    g.write("const long BERS[%d][%d] = {%s};\n"
            % (NPOS, len(STADI),
               ",".join("{%s}" % ",".join("%dL" % bersagli[p][s]
                                          for s in STADI)
                        for p in range(NPOS))))
print("l04cost.h scritto: %d byte di costanti nel banco %d"
      % (2 * (256 + 512 + 256 + NPOS * 8 * 2 + LAYERS * D * 2 + NPOS * D)
         + 4 * NPOS * len(STADI), BANCOK))

# ---------------------------------------------------------- kernel assembly
PTR_LO, PTR_HI = "$f4", "$f5"
ACC_LO, ACC_HI = "$f2", "$f3"
TMP = "$02"


def blocco(j):
    p_lo, p_hi = 2 * j, 2 * j + 1
    base = (p_lo >> 4) * 256
    m_lo = (p_lo & 15) * 16
    m_hi = (p_hi & 15) * 16
    off = " + %d" % base if base else ""
    a = ["        ldy #%d" % j,
         "        lda (%s),y" % PTR_LO,
         "        sta %s" % TMP,
         "        and #15"]
    if m_lo:
        a.append("        ora #%d" % m_lo)
    a += ["        tay", "        clc",
          "        lda %s" % ACC_LO, "        adc PL%s,y" % off,
          "        sta %s" % ACC_LO,
          "        lda %s" % ACC_HI, "        adc PH%s,y" % off,
          "        sta %s" % ACC_HI,
          "        lda %s" % TMP,
          "        lsr", "        lsr", "        lsr", "        lsr"]
    if m_hi:
        a.append("        ora #%d" % m_hi)
    a += ["        tay", "        clc",
          "        lda %s" % ACC_LO, "        adc PL%s,y" % off,
          "        sta %s" % ACC_LO,
          "        lda %s" % ACC_HI, "        adc PH%s,y" % off,
          "        sta %s" % ACC_HI]
    return a


def kernel(bpr, etichetta):
    asm = ["        lda #0", "        sta %s" % ACC_LO, "        sta %s" % ACC_HI]
    for j in range(bpr):
        asm.append("        // byte %d" % j)
        asm += blocco(j)
    asm += ["        lda %s" % ACC_LO, "        sta rowres",
            "        lda %s" % ACC_HI, "        sta rowres + 1",
            "        clc", "        lda %s" % PTR_LO, "        adc #%d" % bpr,
            "        sta %s" % PTR_LO, "        bcc %s" % etichetta,
            "        inc %s" % PTR_HI,
            "%s:" % etichetta]
    return "\n".join(asm)


# ------------------------------------- il MAC 16x8 in assembly (sec. 9-ter)
# Verificato a se' in macq.c (verifica XVII): 81 cicli per MAC contro 560.
# Qui ce ne sono due forme, che differiscono solo per QUALE operando resta
# fisso nell'anello interno - ed e' quello, non la moltiplicazione, a
# decidere la struttura dei dati:
#
#   macq_k  punteggi:    Q fisso per testa, si scorre t.  Cache K a [d][t].
#   macq_v  media pesata: peso[t] fisso, si scorre d.     Cache V a [d][t].
#
# In entrambi i casi l'operando a 16 bit e' costante nell'anello interno,
# quindi i suoi otto puntatori si preparano una volta e non per elemento.
QS1 = [(x * x) // 4 for x in range(512)]
QS2 = [((x - 255) * (x - 255)) // 4 for x in range(512)]
assert max(QS1) < 65536 and max(QS2) < 65536


def bytes_di(v, alto):
    return ",".join(str((x >> 8) & 255 if alto else x & 255) for x in v)


# ⚠️⚠️ PAGINA ZERO.  I puntatori NON possono stare a $02-$11: WORK di oscar64
# occupa $03-$0B, e non e' scratch da usare fra una chiamata e l'altra - il
# compilatore ci tiene valori VIVI attraverso una chiamata di funzione.  Un
# assembly inline che ci scrive dentro gli distrugge le variabili sotto i
# piedi, e il programma si blocca senza dire niente.  Stessa lezione di
# sec. 11, dall'altro verso: li' era l'ottimizzatore a invadere la zona
# dell'assembly, qui e' l'assembly a invadere quella del compilatore.
#   $03-$0B  WORK      $19-$1a IP    $1b-$1e ACCU   $1f-$22 ADDR
#   $23-$24  SP        $F2-$F5 kernel ternario      $F7-$FF globali con -Oz
PZ_PTR = 0xE0      # 16 byte: gli otto puntatori
PZ_PLO = 0xF0
PZ_PHI = 0xF1


def puntatori(sorgente, d):
    """copia in pagina zero gli otto puntatori dell'operando a 16 bit"""
    a = []
    for k in range(8):
        a += ["        lda %s + %d" % (sorgente, d * 16 + 2 * k),
              "        sta $%02x" % (PZ_PTR + 2 * k),
              "        lda %s + %d" % (sorgente, d * 16 + 2 * k + 1),
              "        sta $%02x" % (PZ_PTR + 1 + 2 * k)]
    return a


def prodotto(pl, ph, a0, a1, a2, etichetta, indicizzato):
    """un prodotto senza segno accumulato su 24 bit"""
    ix = ",x" if indicizzato else ""
    return ["        sec",
            "        lda ($%02x),y" % pl, "        sbc ($%02x),y" % (pl + 2),
            "        sta $%02x" % PZ_PLO,
            "        lda ($%02x),y" % ph, "        sbc ($%02x),y" % (ph + 2),
            "        sta $%02x" % PZ_PHI,
            "        clc",
            "        lda %s%s" % (a0, ix), "        adc $%02x" % PZ_PLO,
            "        sta %s%s" % (a0, ix),
            "        lda %s%s" % (a1, ix), "        adc $%02x" % PZ_PHI,
            "        sta %s%s" % (a1, ix),
            "        bcc %s" % etichetta,
            "        inc %s%s" % (a2, ix),
            "%s:" % etichetta]


asm_k = []
for d in range(KVDIM):
    asm_k.append("        // ---- punteggi, d = %d ----" % d)
    asm_k += puntatori("PTRK", d)
    asm_k += ["        ldx XBASE",
              "lpk%d:" % d,
              "        lda KU + %d,x" % (d * NT2),
              "        tay"]
    asm_k += prodotto(PZ_PTR, PZ_PTR + 4, "ACH0", "ACH1", "ACH2", "nhk%d" % d, True)
    asm_k += prodotto(PZ_PTR + 8, PZ_PTR + 12, "ACL0", "ACL1", "ACL2", "nlk%d" % d, True)
    asm_k += ["        inx", "        cpx NTC", "        bne lpk%d" % d]

# macq_v: X e' fisso (la cella di cache), l'anello sui d e' tutto srotolato,
# quindi gli accumulatori si indirizzano in assoluto senza indice.
asm_v = ["        ldx XCUR"]
asm_v += puntatori("PTRV", 0)
for d in range(KVDIM):
    asm_v.append("        // ---- media pesata, d = %d ----" % d)
    asm_v.append("        ldy VU + %d,x" % (d * NT2))
    asm_v += prodotto(PZ_PTR, PZ_PTR + 4, "VH0 + %d" % d, "VH1 + %d" % d,
                      "VH2 + %d" % d, "nhv%d" % d, False)
    asm_v += prodotto(PZ_PTR + 8, PZ_PTR + 12, "VL0 + %d" % d, "VL1 + %d" % d,
                      "VL2 + %d" % d, "nlv%d" % d, False)

# macoh: output head, UNA dimensione contro tutti i 512 token.
# Stessa forma di macq_k - operando a 16 bit fisso, 8 bit contiguo - ma la
# riga e' di 512 byte e X ne indirizza 256, quindi due meta' con offset
# costante.  Il ciclo si chiude su  inx / bne: 256 giri esatti.
# ⚠️ Non srotolato sulle dimensioni: sarebbero 128 blocchi, decine di KB.
# Qui il costo della preparazione dei puntatori si ammortizza su 512
# elementi invece che su 16, quindi non serve.
asm_o = puntatori("PTRO", 0)
for meta in (0, 256):
    asm_o.append("        // ---- token %d-%d ----" % (meta, meta + 255))
    asm_o += ["        ldx #0", "lpo%d:" % meta,
              "        lda ROW + %d,x" % meta, "        tay"]
    asm_o += prodotto(PZ_PTR, PZ_PTR + 4,
                      "OHH0 + %d" % meta, "OHH1 + %d" % meta,
                      "OHH2 + %d" % meta, "nho%d" % meta, True)
    asm_o += prodotto(PZ_PTR + 8, PZ_PTR + 12,
                      "OHL0 + %d" % meta, "OHL1 + %d" % meta,
                      "OHL2 + %d" % meta, "nlo%d" % meta, True)
    asm_o += ["        inx", "        bne lpo%d" % meta]

MACQ_C = """/* ---- banco %d: il MAC 16x8 e le sue tabelle (sec. 9-ter) ---- */
#pragma code ( bcode%d )
#pragma data ( bdata%d )

/* f(x) = x*x/4.  a*b = f(a+b) - f(a-b), solo senza segno.
   ⚠️ #pragma align DOPO la dichiarazione, se no error 3005. */
__export const unsigned char T1L[512] = {%s};
#pragma align( T1L, 256 )
__export const unsigned char T1H[512] = {%s};
#pragma align( T1H, 256 )
__export const unsigned char T2L[512] = {%s};
#pragma align( T2L, 256 )
__export const unsigned char T2H[512] = {%s};
#pragma align( T2H, 256 )

__noinline void macq_k(void)
{
    __asm
    {
%s
    }
}

__noinline void macq_v(void)
{
    __asm
    {
%s
    }
}

__noinline void macoh(void)
{
    __asm
    {
%s
    }
}
""" % (BANCOM, BANCOM, BANCOM,
       bytes_di(QS1, False), bytes_di(QS1, True),
       bytes_di(QS2, False), bytes_di(QS2, True),
       "\n".join(asm_k), "\n".join(asm_v), "\n".join(asm_o))

# ------------------------------------------- sezioni dei banchi e descrittori
NOMI = {}     # idx -> simbolo C
pezzi = []
for banco in BANCHI:
    dentro = sorted(i for i in piano if piano[i][0] == banco)
    pezzi.append("/* ---- banco %d: kernel %d byte/riga + %s ---- */"
                 % (banco, banco_bpr[banco],
                    ", ".join("L%d_%s" % (i // 7, ORDINE[i % 7])
                              for i in dentro)))
    pezzi.append("#pragma code ( bcode%d )" % banco)
    pezzi.append("#pragma data ( bdata%d )" % banco)
    for i in dentro:
        _, nbyte, _, fn = piano[i]
        nome = "M_L%d_%s" % (i // 7, ORDINE[i % 7])
        NOMI[i] = nome
        pezzi.append("__export const char %s[%d] = {" % (nome, nbyte))
        pezzi.append("#embed %d 0 \"%s\"" % (nbyte, fn))
        pezzi.append("};")
    pezzi.append("__noinline void macrow_b%d(void)" % banco)
    pezzi.append("{")
    pezzi.append("    __asm")
    pezzi.append("    {")
    pezzi.append(kernel(banco_bpr[banco], "av_b%d" % banco))
    pezzi.append("    }")
    pezzi.append("}")
    pezzi.append("")

pezzi.append(MACQ_C)

# I quattro banchi dell'embedding trasposto.  Solo dati: il codice
# dell'output head gira in RAM e commuta, perche' una passata tocca tutti e
# quattro i banchi (sec. 13).
pezzi.append("/* ---- banchi %d-%d: embedding TRASPOSTO e spostato di +128,"
             " %d dimensioni per banco ---- */" % (BANCOE, BANCOE + 3, DPB))
for i in range(4):
    pezzi.append("#pragma data ( bdataE%d )" % i)
    pezzi.append("__export const char EMBT%d[%d] = {" % (i, BANCO))
    pezzi.append("#embed %d 0 \"embT%d.bin\"" % (BANCO, i))
    pezzi.append("};")
pezzi.append("#pragma data ( data )")
pezzi.append("")

pezzi.append("/* ---- banco %d: le costanti, ~9 KB che in RAM non ci stanno "
             "---- */" % BANCOK)
pezzi.append("#pragma data ( bdatak )")
pezzi.append("#include \"l04cost.h\"")
pezzi.append("")
pezzi.append("#pragma code ( code )")
pezzi.append("#pragma data ( data )")
BANCHI_C = "\n".join(pezzi)

# Le regioni si emettono qui e non si include banchi.h: quel file dichiara
# anche i quattro banchi dell'embedding, che qui non servono.  Il PIANO pero'
# viene da li', quindi non c'e' modo di divergere.
REGIONI = "\n".join(
    "#pragma section( bcode%d, 0 )\n"
    "#pragma section( bdata%d, 0 )\n"
    "#pragma region( bank%d, 0x8000, 0xc000, , %d, { bcode%d, bdata%d } )"
    % (b, b, b, b, b, b) for b in BANCHI)
REGIONI += ("\n#pragma section( bdatak, 0 )\n"
            "#pragma region( bankk, 0x8000, 0xc000, , %d, { bdatak } )"
            % BANCOK)
REGIONI += ("\n#pragma section( bcode%d, 0 )\n"
            "#pragma section( bdata%d, 0 )\n"
            "#pragma region( bankm, 0x8000, 0xc000, , %d, "
            "{ bcode%d, bdata%d } )"
            % (BANCOM, BANCOM, BANCOM, BANCOM, BANCOM))
for i in range(4):
    REGIONI += ("\n#pragma section( bdataE%d, 0 )\n"
                "#pragma region( bankE%d, 0x8000, 0xc000, , %d, "
                "{ bdataE%d } )" % (i, i, BANCOE + i, i))

# descrittori in RAM: 28 puntatori, 28 banchi, 28 conteggi di riga
RIGHE = []
for i in range(LAYERS * 7):
    _, nbyte, bpr, _ = piano[i]
    RIGHE.append(nbyte // bpr)
DESCR = (
    "const char * const MPTR[%d] = {%s};\n" % (
        LAYERS * 7, ",".join(NOMI[i] for i in range(LAYERS * 7))) +
    "const unsigned char MBANCO[%d] = {%s};\n" % (
        LAYERS * 7, ",".join(str(piano[i][0]) for i in range(LAYERS * 7))) +
    "const int MRIGHE[%d] = {%s};\n" % (LAYERS * 7, ",".join(map(str, RIGHE))) +
    "const char * const EMBP[4] = { EMBT0, EMBT1, EMBT2, EMBT3 };")

SWITCH = "\n".join(
    "        case %d:  macrow_b%d(); break;" % (b, b) for b in BANCHI[:-1]
) + "\n        default: macrow_b%d(); break;" % BANCHI[-1]

# ------------------------------------------------------------------- l04.c
sorgente = r'''/* l04.c - GENERATO DA gen_l04.py.  QUATTRO LAYER IN CASCATA sul C64,
   kernel MAC in assembly dai banchi, cache KV a int8, costanti in cartuccia.

   Verifica 17 stadi per posizione su NPOS posizioni: x0 piu' xb, q, cat, x2
   di ciascuno dei quattro layer.  Bersagli da oracolo_att.exe -dump4.

   ⚠️ REGOLA DEL BANCO: ogni funzione che legge una costante mette
      eflash.bank = BANCOK come prima cosa; matvec commuta sul banco dei suoi
      pesi.  Nessuno ripristina niente: chi legge commuta.

   ⚠️ COMPILARE CON -O2, NON -O3 (sec. 11: -Oz e la pagina zero).

   Compilare ed eseguire:  .\c64.ps1 l04 -Cicli 250000000
*/

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include "l04data.h"

#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, stack } )
#pragma stacksize( 2048 )

__REGIONI__

#define DIM    128
#define HID    352
#define LAYERS 4
#define QB     10
#define HEADS  8
#define HDIM   16
#define KVD    16
#define KSH    7
#define VSH    5

/* condivise fra RAM e banchi */
unsigned char PL[2816];
unsigned char PH[2816];
int rowres;

/* CACHE KV — la forma la detta il MAC in assembly (sec. 9-ter), non il
   comodo del C:

     - TRASPOSTA a [d][cella]: cosi' l'anello interno scorre le posizioni con
       l'operando a 16 bit FISSO, e i suoi otto puntatori si preparano una
       volta per d invece che per elemento.  La scrittura diventa sparsa -
       16 byte sparpagliati per posizione - contro 16 letture contigue per
       testa: si paga una volta e si guadagna otto.
     - la cella e' l*NPOS + t, cosi' layer e posizione insieme STANNO IN UN
       BYTE e vanno nel registro X: nessun puntatore da ricalcolare, nessuna
       copia di appoggio.  A SEQ 64 sono 4*64 = 256 celle, il massimo che X
       regge - e non e' una coincidenza fortunata, e' il vincolo che ha
       deciso la disposizione.
     - memorizzata SPOSTATA di +128, cosi' e' senza segno e la tabella dei
       quarti di quadrato basta com'e'.  Resta un byte per valore: il
       bilancio RAM di sec. 14 non cambia.

   ⚠️ Tutto quello che l'assembly nomina va dichiarato QUI, prima delle
   sezioni dei banchi: piu' sotto oscar64 da' "error 3027: Undefined label". */
unsigned char KU[KVD][NT2];
unsigned char VU[KVD][NT2];
unsigned      KSUM[NT2];      /* somma_d ku[d][cella], calcolata scrivendo */

/* accumulatori a 24 bit come tre array di byte paralleli: si indicizzano
   con X senza doverli moltiplicare per tre */
unsigned char ACH0[NT2], ACH1[NT2], ACH2[NT2];
unsigned char ACL0[NT2], ACL1[NT2], ACL2[NT2];
unsigned char VH0[KVD], VH1[KVD], VH2[KVD];
unsigned char VL0[KVD], VL1[KVD], VL2[KVD];

unsigned      PTRK[KVD][8];   /* otto puntatori per ogni d, una volta per testa */
unsigned      PTRV[8];        /* otto puntatori di peso[t], una volta per t */
unsigned char XBASE, NTC, XCUR;

/* OUTPUT HEAD.  512 logit, uno per token del vocabolario, ciascuno un
   prodotto scalare di 128 termini contro xb: 65536 MAC, piu' di tutto il
   resto del token se fatti in C (~36 M cicli contro ~6).
   L'embedding trasposto sta in quattro banchi e il codice in RAM, perche'
   una passata li tocca tutti e quattro (sec. 13).  ROW e' la riga della
   dimensione corrente, copiata dal banco: serve perche' l'indirizzo di
   partenza dipende da d, e in assoluto indicizzato la base deve essere
   costante. */
unsigned char ROW[VOC];
unsigned char OHH0[VOC], OHH1[VOC], OHH2[VOC];
unsigned char OHL0[VOC], OHL1[VOC], OHL2[VOC];
unsigned      PTRO[8];

__BANCHI__

__DESCR__

int x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], cat[DIM], o[DIM];
int h1[HID], h3[HID];
int xq[HID];
long att[NPOS];

static CharWin cw;

static void put(const char * s) { cwin_put_string(&cw, s, 7); }
static void nl(void)            { cwin_cursor_newline(&cw); }

static void putnum(long v)
{
    char buf[12];
    char i = 12;
    unsigned long u;
    if (v < 0) { cwin_put_char(&cw, p'-', 7); u = (unsigned long)(-v); }
    else       u = (unsigned long)v;
    if (u == 0) { cwin_put_char(&cw, p'0', 7); return; }
    while (u) { buf[--i] = p'0' + (char)(u % 10); u /= 10; }
    while (i < 12) cwin_put_char(&cw, buf[i++], 7);
}

/* ------------------------------------------------------- cronometro CIA 2 */
static volatile byte scarto;
static unsigned giri;

static void tmr_start(void)
{
    cia2.crb = 0x00;
    cia2.tb  = 0xffff;
    scarto = cia2.icr;
    giri = 0;
    cia2.crb = 0x11;
}

static void tmr_poll(void)
{
    if (cia2.icr & 0x02) giri++;
}

static unsigned long tmr_stop(void)
{
    unsigned resto;
    cia2.crb = 0x00;
    tmr_poll();
    resto = 0xffff - cia2.tb;
    return (unsigned long)giri * 65536UL + resto;
}

/* c_alt e' la voce che nessuna misura precedente aveva mai isolato:
   RMSnorm, RoPE, SwiGLU, residue.

   ⚠️ LE tmr_poll NON SONO DECORATIVE.  Il timer B e' a 16 bit e tmr_stop
   rileva UN SOLO giro: se fra due interrogazioni passano piu' di 131072
   cicli, ogni giro oltre il primo sparisce e la misura torna un RESTO
   modulo 65536, cioe' un numero piccolo e plausibile.  E' successo davvero:
   con una sola tmr_poll per testa l'attenzione risultava piatta a ~2,2 M
   per token da posizione 2 in poi, il che avrebbe voluto dire che il ciclo
   sui t non cresce - assurdo.  Regola: interrogare dentro il ciclo piu'
   interno che possa superare i 65536 cicli.  Le poll costano ~40 cicli e
   stanno DENTRO la misura, quindi la gonfiano di poco: mai il contrario. */
static unsigned long c_build = 0, c_mac = 0, c_alt = 0;

/* L'attenzione e' l'unica voce che CRESCE COL CONTESTO: a posizione p costa
   in proporzione a p+1.  Misurarla dentro il "resto" darebbe un numero vero
   solo per frasi corte, quindi si tiene separata, e si tiene anche il costo
   all'ULTIMA posizione, che e' quello da estrapolare a SEQ 64. */
static unsigned long c_att = 0, c_oh = 0;
static unsigned long c_attp[NPOS];    /* costo per posizione, 4 layer */
static int GEN[NPOS];                 /* i token generati, uno per posizione */
static long R0MAC, R0RIF;             /* logit del token 0 a pos 0: MAC e C */

/* -------------------------------------------------------------- verifiche */
static int  errori = 0, confronti = 0;
static int  pr_pos = -1, pr_st = -1;
static long pr_ott = 0, pr_att = 0;

static void verifica(int pos, int st, long s)
{
    eflash.bank = BANCOK;
    confronti++;
    if (s != BERS[pos][st]) {
        errori++;
        if (pr_pos < 0) {
            pr_pos = pos; pr_st = st;
            pr_ott = s;   pr_att = BERS[pos][st];
        }
    }
}

static void verifica_vec(int pos, int st, const int * v, int n)
{
    long s = 0;
    int i;
    for (i = 0; i < n; i++) s += (long)v[i];
    verifica(pos, st, s);
}

static char quant_shift(const int * v, int n)
{
    long mx = 0;
    long a;
    int i;
    char s = 0;
    for (i = 0; i < n; i++) {
        a = (long)v[i];
        if (a < 0) a = -a;
        if (a > mx) mx = a;
    }
    while ((mx >> s) > 127L) s++;
    for (i = 0; i < n; i++) xq[i] = v[i] >> s;
    return s;
}

static void build(int ncoppie)
{
    unsigned char * pl = PL;
    unsigned char * ph = PH;
    const int * xp = xq;
    int p, xa, xbb, sm, df;
    for (p = 0; p < ncoppie; p++) {
        xa  = xp[0];
        xbb = xp[1];
        sm = xa + xbb;
        df = xa - xbb;
        pl[10] = (unsigned char)sm;   ph[10] = (unsigned char)(sm >> 8);
        sm = -sm;
        pl[0]  = (unsigned char)sm;   ph[0]  = (unsigned char)(sm >> 8);
        pl[2]  = (unsigned char)df;   ph[2]  = (unsigned char)(df >> 8);
        df = -df;
        pl[8]  = (unsigned char)df;   ph[8]  = (unsigned char)(df >> 8);
        pl[6]  = (unsigned char)xa;   ph[6]  = (unsigned char)(xa >> 8);
        xa = -xa;
        pl[4]  = (unsigned char)xa;   ph[4]  = (unsigned char)(xa >> 8);
        pl[9]  = (unsigned char)xbb;  ph[9]  = (unsigned char)(xbb >> 8);
        xbb = -xbb;
        pl[1]  = (unsigned char)xbb;  ph[1]  = (unsigned char)(xbb >> 8);
        pl[5]  = 0;                   ph[5]  = 0;
        xp += 2;
        pl += 16;
        ph += 16;
    }
}

/* m = indice della matrice, l*7 + k con k in {wq,wk,wv,wo,w1,w2,w3}.
   Il banco identifica il kernel: nel piano nessun banco mescola larghezze. */
/* ⚠️ L'EPILOGO DI RIGA e' caro quanto un terzo del kernel: 1263 cicli per
   riga misurati (sec. 7).  Le due voci sono la moltiplicazione 32x32 e
   soprattutto lo SHIFT DI AMPIEZZA VARIABILE, che oscar64 fa con un ciclo
   di un giro per bit - e sh vale 9-15.
   La parte a byte interi si stacca: per lo shift aritmetico
       (t >> 8) >> k  ==  t >> (8 + k)
   quindi si fa un >>8 a costante, che e' uno spostamento di byte, e resta
   un residuo di al piu' 7 bit.  Non cambia un solo risultato. */
static void matvec(int m, int * y, char sx)
{
    int i, righe;
    long t, mant;
    char banco = MBANCO[m];
    char sh = SHIFTM[m] - sx;
    char pre8 = 0;

    if (sh >= 8) { pre8 = 1; sh = sh - 8; }

    /* ⚠️ PROVATO E SCARTATO: spezzare la mantissa in due byte senza segno per
       prendere mul32by8 al posto della 32x32,
           rowres*mant = 256*(rowres*mh) + rowres*ml
       e' esatto (336 confronti a zero errori) ma COSTA DI PIU': 19.858.134
       contro 19.679.556.  Due mul32by8 piu' uno shift e una somma battono
       una mul32 solo quando sostituiscono la moltiplicazione, non quando la
       raddoppiano - identica lezione del tentativo (a) di sec. 9-bis. */
    righe = MRIGHE[m];
    mant  = (long)MANTM[m];
    eflash.bank = banco;
    *(const char **)0x00F4 = MPTR[m];

    for (i = 0; i < righe; i++) {
        switch (banco) {
__SWITCH__
        }
        t = (long)rowres;
        t = t * mant;
        if (pre8) t = t >> 8;
        t = t >> sh;
        if (t >  32767L) t =  32767L;
        if (t < -32768L) t = -32768L;
        y[i] = (int)t;
        tmr_poll();
    }
}

static void rmsnorm(int * out, const int * v, const int * w, int n)
{
    long ss = 0;
    long t, inv, vi;
    int i, e = 0, k;
    eflash.bank = BANCOK;
    for (i = 0; i < n; i++) {
        vi = (long)v[i];
        ss += vi * vi;
    }
    tmr_poll();
    ss = ss / n;
    if (ss < 1) ss = 1;
    while (ss <  1048576L) { ss = ss << 2; e++; }
    while (ss >= 4194304L) { ss = ss >> 2; e--; }
    t = ss - 1048576L;
    t = t * 256L;
    t = t / 3145728L;
    k = (int)t;
    if (k > 255) k = 255;
    if (k < 0)   k = 0;
    inv = (long)RSQ[k];
    inv = inv << e;
    for (i = 0; i < n; i++) {
        t = (long)v[i];
        t = t * inv;
        t = t >> 14;
        t = t * (long)w[i];
        t = t >> 12;
        if (t >  32767L) t =  32767L;
        if (t < -32768L) t = -32768L;
        out[i] = (int)t;
        if ((i & 63) == 0) tmr_poll();
    }
}

/* RoPE: ruota le coppie ADIACENTI (0,1) (2,3) ... con le frequenze 0..7 */
static void rope(int * p, int pos)
{
    int i;
    long a, bq, r, im;
    eflash.bank = BANCOK;
    for (i = 0; i < 8; i++) {
        r  = (long)p[2 * i];
        im = (long)p[2 * i + 1];
        a  = r * (long)RCOS[pos][i];
        bq = im * (long)RSIN[pos][i];
        a  = a - bq;
        a  = a >> 14;
        if (a >  32767L) a =  32767L;
        if (a < -32768L) a = -32768L;
        bq = r * (long)RSIN[pos][i];
        r  = im * (long)RCOS[pos][i];
        bq = bq + r;
        bq = bq >> 14;
        if (bq >  32767L) bq =  32767L;
        if (bq < -32768L) bq = -32768L;
        p[2 * i]     = (int)a;
        p[2 * i + 1] = (int)bq;
    }
}

/* MQA: una sola testa KV, condivisa dalle otto.  Un reciproco per testa.

   ⚠️ NIENTE long * long QUI DENTRO.  Scritto come
       s += (long)q[...] * (long)KC[...];
   il C chiede la moltiplicazione 32x32 del runtime, che costa 908 cicli per
   MAC contro i 21 del kernel ternario: a contesto pieno erano 59,7 M cicli
   per token, il 71% del tempo (sec. 9-bis).

   TRE STRADE PROVATE IN C, MISURATE.  Non rifarle:

     a) spezzare in due prodotti a 16 bit,
        q*k = 256*((q>>8)*k) + ((q&255)*k),  con array di int.
        Esatto (272/0) ma PEGGIORA: 1086k per posizione contro 930k.
        In oscar64  int * int  NON e' una tabella in linea, e' un JSR mul16,
        e due mul16 piu' due somme a 32 bit costano piu' di un mul32.

     b) la stessa con array di char, per far scegliere la 8x8 nativa.
        Sceglie mul16by8, ed e' la piu' veloce di tutte... e SBAGLIATA:
        224 errori su 272.  Vedi la trappola qui sotto.

     c) ⭐ QUELLA CHE SI USA: un solo prodotto per elemento, con l'operando
        piccolo reso UNSIGNED CHAR, per cui oscar64 sceglie mul32by8.
            k + 128  sta in [0, 255]
            somma(q*k) = somma(q*(k+128)) - 128*somma(q)
        somma(q) sulla testa si calcola una volta sola e i due termini con
        128 si elidono: non cambia un bit.  573k per posizione, 560 cicli
        per MAC contro 908.  Margini: q*(k+128) <= 32767*255 = 8,4 M, per 64
        addendi 535 M, dentro i 2,1 G di un int32.

   ⚠️⚠️ TRAPPOLA, LA PEGGIORE DI QUESTA SESSIONE.  In (b),

       acch += (long)(qhi[d] * kp[d]);          // unsigned char * signed char
       acch += (long)((int)qhi[d] * (int)kp[d]);

   danno RISULTATI DIVERSI.  In C sono la stessa espressione - entrambi gli
   operandi si promuovono a int comunque - e infatti la seconda da' 272/0
   mentre la prima da' 224 errori: senza i cast il segno dell'operando a 8
   bit si perde.  E la versione sbagliata era la piu' VELOCE, cioe' quella
   che uno terrebbe.  Regola: **con operandi a 8 bit di segno diverso,
   mettere i cast (int) espliciti anche dove il C dice che sono inutili**, e
   riverificare sempre i 272 confronti dopo aver cambiato un tipo.

   d) ⭐ QUELLO CHE SI USA ORA: il MAC 16x8 in assembly (sec. 9-ter,
      verifica XVII), 81 cicli contro 560.  Il C qui sotto non moltiplica
      piu': prepara i puntatori, chiama, e ricombina.

   Le due formule, entrambe esatte per costruzione, con  ku = k + 128  e
   vu = v + 128  gia' memorizzati spostati:

     punteggio(t) = 256*( somma_d qhu[d]*ku[d][t] - 128*Ksum[t] )
                        +   somma_d  ql[d]*ku[d][t]  -  128*Sq
     uscita(d)    = 256*  somma_t  ph[t]*vu[d][t]
                        +   somma_t  pl[t]*vu[d][t]  -  128*Sp

   Ksum si calcola SCRIVENDO la riga di cache, Sq e Sp una volta per testa:
   dentro l'anello non resta nessuna correzione e nessun prodotto con segno.
   Sul lato dei pesi non serve nemmeno lo spostamento, perche' peso[t] e' gia'
   in [0, 32768] e spezzato in due byte da' due operandi senza segno. */
static void attenzione(int l, int pos)
{
    int h, d, t;
    long s, best, somma, inv;
    long sq, sp, ach, acl;
    int qq;
    unsigned char qhu, ql, base, x;
    unsigned pw;

    base = (unsigned char)(l * NPOS);

    for (h = 0; h < HEADS; h++) {

        /* --- punteggi: Q fisso, si scorre la cache --- */
        sq = 0;
        for (d = 0; d < HDIM; d++) {
            qq  = q[h * HDIM + d];
            sq += (long)qq;
            qhu = (unsigned char)((qq >> 8) + 128);
            ql  = (unsigned char)(qq & 255);
            PTRK[d][0] = (unsigned)T1L + qhu;
            PTRK[d][1] = (unsigned)T2L + (unsigned char)(qhu ^ 255);
            PTRK[d][2] = (unsigned)T1H + qhu;
            PTRK[d][3] = (unsigned)T2H + (unsigned char)(qhu ^ 255);
            PTRK[d][4] = (unsigned)T1L + ql;
            PTRK[d][5] = (unsigned)T2L + (unsigned char)(ql ^ 255);
            PTRK[d][6] = (unsigned)T1H + ql;
            PTRK[d][7] = (unsigned)T2H + (unsigned char)(ql ^ 255);
        }
        sq = sq << 7;

        for (t = 0; t <= pos; t++) {
            x = base + (unsigned char)t;
            ACH0[x] = 0; ACH1[x] = 0; ACH2[x] = 0;
            ACL0[x] = 0; ACL1[x] = 0; ACL2[x] = 0;
        }
        XBASE = base;
        NTC   = base + (unsigned char)pos + 1;
        eflash.bank = BANCOM;
        macq_k();
        tmr_poll();

        eflash.bank = BANCOK;
        best = -2147483647L;
        for (t = 0; t <= pos; t++) {
            x = base + (unsigned char)t;
            ach = (long)ACH0[x] + ((long)ACH1[x] << 8) + ((long)ACH2[x] << 16);
            acl = (long)ACL0[x] + ((long)ACL1[x] << 8) + ((long)ACL2[x] << 16);
            ach = ach - ((long)KSUM[x] << 7);   /* NON 128L*..., che e' una mul32 */
            s = (ach << 8) + acl - sq;
            s = s >> (QB + 2 - KSH);
            att[t] = s;
            if (s > best) best = s;
            tmr_poll();
        }

        somma = 0;
        for (t = 0; t <= pos; t++) {
            s = best - att[t];
            s = s >> 4;
            if (s > 511L) s = 511L;
            att[t] = (long)TEXP[(int)s];
            somma += att[t];
        }
        if (somma < 1) somma = 1;
        inv = 1073741824L / somma;

        /* --- media pesata: peso[t] fisso, si scorrono le dimensioni --- */
        for (d = 0; d < HDIM; d++) {
            VH0[d] = 0; VH1[d] = 0; VH2[d] = 0;
            VL0[d] = 0; VL1[d] = 0; VL2[d] = 0;
        }
        sp = 0;
        for (t = 0; t <= pos; t++) {
            s = att[t] * inv;
            s = s >> 15;
            sp += s;
            pw = (unsigned)s;
            qhu = (unsigned char)(pw >> 8);      /* <= 128, gia' senza segno */
            ql  = (unsigned char)(pw & 255);
            PTRV[0] = (unsigned)T1L + qhu;
            PTRV[1] = (unsigned)T2L + (unsigned char)(qhu ^ 255);
            PTRV[2] = (unsigned)T1H + qhu;
            PTRV[3] = (unsigned)T2H + (unsigned char)(qhu ^ 255);
            PTRV[4] = (unsigned)T1L + ql;
            PTRV[5] = (unsigned)T2L + (unsigned char)(ql ^ 255);
            PTRV[6] = (unsigned)T1H + ql;
            PTRV[7] = (unsigned)T2H + (unsigned char)(ql ^ 255);
            XCUR = base + (unsigned char)t;
            eflash.bank = BANCOM;
            macq_v();
            tmr_poll();
        }
        sp = sp << 7;

        for (d = 0; d < HDIM; d++) {
            ach = (long)VH0[d] + ((long)VH1[d] << 8) + ((long)VH2[d] << 16);
            acl = (long)VL0[d] + ((long)VL1[d] << 8) + ((long)VL2[d] << 16);
            s = (ach << 8) + acl - sp;
            s = s >> (15 - VSH);
            if (s >  32767L) s =  32767L;
            if (s < -32768L) s = -32768L;
            cat[h * HDIM + d] = (int)s;
        }
        eflash.bank = BANCOK;
    }
}

/* OUTPUT HEAD.  logit(t) = somma_d xb[d] * emb[t][d], per tutti i 512 token.
   Con l'embedding memorizzato trasposto e spostato (eu = e + 128):

     logit(t) = 256*( somma_d xbhu[d]*eu[d][t] - 128*Esum[t] )
                    +   somma_d  xbl[d]*eu[d][t]
                    -   128*Sxb

   Esum[t] dipende solo dal modello ed e' precalcolato; Sxb una volta per
   passata.  Restituisce l'argmax e lascia in *somma la somma dei logit,
   che e' il bersaglio buono: prende tutti e 512 i valori, quindi un solo
   numero sbagliato la sposta. */
static long LG0;      /* il logit del token 0, per il bersaglio stretto */
static long LG0RIF;   /* lo stesso, calcolato in C sugli STESSI byte */

static int output_head(long * somma)
{
    int d, t, best;
    long sxb, lg, ach, acl, bestv;
    unsigned char xbhu, xbl;
    const char * src;
    int qq;

    for (t = 0; t < VOC; t++) {
        OHH0[t] = 0; OHH1[t] = 0; OHH2[t] = 0;
        OHL0[t] = 0; OHL1[t] = 0; OHL2[t] = 0;
    }
    sxb = 0;
    for (d = 0; d < DIM; d++) sxb += (long)xb[d];
    sxb = sxb << 7;

    LG0RIF = 0;
    for (d = 0; d < DIM; d++) {
        /* la riga della dimensione d sta nel banco BANCOE + d/DPB, e non e'
           mai a cavallo di due banchi: DPB righe da VOC byte riempiono un
           banco esatto (verificato in export_banchi.py) */
        eflash.bank = BANCOE + (d / DPB);
        src = EMBP[d / DPB] + (long)(d % DPB) * VOC;
        /* la copia in C e' gia' a ~16 cicli per byte: la stessa cosa scritta
           in assembly non ha guadagnato niente (misurato), quindi resta qui */
        for (t = 0; t < VOC; t++) ROW[t] = (unsigned char)src[t];

        /* riferimento in C sugli STESSI byte appena copiati, per il solo
           token 0: se questo torna e il MAC no, il difetto e' nel MAC;
           se non torna nemmeno questo, e' nel dato o nel banco. */
        LG0RIF += (long)xb[d] * (long)((int)ROW[0] - 128);

        qq   = xb[d];
        xbhu = (unsigned char)((qq >> 8) + 128);
        xbl  = (unsigned char)(qq & 255);
        PTRO[0] = (unsigned)T1L + xbhu;
        PTRO[1] = (unsigned)T2L + (unsigned char)(xbhu ^ 255);
        PTRO[2] = (unsigned)T1H + xbhu;
        PTRO[3] = (unsigned)T2H + (unsigned char)(xbhu ^ 255);
        PTRO[4] = (unsigned)T1L + xbl;
        PTRO[5] = (unsigned)T2L + (unsigned char)(xbl ^ 255);
        PTRO[6] = (unsigned)T1H + xbl;
        PTRO[7] = (unsigned)T2H + (unsigned char)(xbl ^ 255);

        eflash.bank = BANCOM;
        macoh();
        tmr_poll();
    }

    eflash.bank = BANCOK;
    best  = 0;
    bestv = -2147483647L;
    lg    = 0;
    for (t = 0; t < VOC; t++) {
        ach = (long)OHH0[t] + ((long)OHH1[t] << 8) + ((long)OHH2[t] << 16);
        acl = (long)OHL0[t] + ((long)OHL1[t] << 8) + ((long)OHL2[t] << 16);
        /* shift a 32 bit in un passaggio separato: in espressione composta
           oscar64 non lo applica (trappola 1 del documento) */
        ach = ach - ((long)ESUM[t] << 7);   /* NON 128L*..., che e' una mul32 */
        ach = ach << 8;
        ach = ach + acl;
        ach = ach - sxb;
        if (t == 0) LG0 = ach;
        lg += ach;
        if (ach > bestv) { bestv = ach; best = t; }
        tmr_poll();
    }
    *somma = lg;
    return best;
}

int main(void)
{
    int i, l, m, pos, d;
    char s;
    long t;
    unsigned long tot, cc;
    unsigned char cella;
    unsigned ksum;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"4 layer, "); putnum((long)NPOS); put(p" posizioni"); nl();

    for (pos = 0; pos < NPOS; pos++) {

        /* segna dove si e' arrivati: se il tetto di cicli scade a meta', lo
           schermo dice se il programma e' lento o fermo */
        putnum((long)pos); cwin_put_char(&cw, p' ', 7);

        tmr_start();
        eflash.bank = BANCOK;
        for (i = 0; i < DIM; i++) {
            t = (long)EMBT[pos][i];
            t = t * EMB_MANT;
            t = t >> (EMB_SHIFT - QB);
            if (t >  32767L) t =  32767L;
            if (t < -32768L) t = -32768L;
            x[i] = (int)t;
            if ((i & 63) == 0) tmr_poll();
        }
        c_alt += tmr_stop();
        verifica_vec(pos, 0, x, DIM);

        for (l = 0; l < LAYERS; l++) {
            m = l * 7;

            tmr_start(); rmsnorm(xb, x, NRM_AN[l], DIM); c_alt += tmr_stop();
            verifica_vec(pos, 1 + l * 4 + 0, xb, DIM);

            s = quant_shift(xb, DIM);
            tmr_start(); build(64);          c_build += tmr_stop();
            tmr_start();
            matvec(m + 0, q,  s);
            matvec(m + 1, kk, s);
            matvec(m + 2, vv, s);
            c_mac += tmr_stop();
            verifica_vec(pos, 1 + l * 4 + 1, q, DIM);

            tmr_start();
            for (i = 0; i < HEADS; i++) rope(q + i * HDIM, pos);
            rope(kk, pos);

            /* Cache int8, ARROTONDANDO (sec. 14).  Scrittura nella forma che
               vuole il MAC: trasposta, spostata di +128, con la somma di
               riga calcolata qui una volta per tutte le letture. */
            cella = (unsigned char)(l * NPOS + pos);
            ksum = 0;
            for (d = 0; d < KVD; d++) {
                t = (long)kk[d] + 64L;
                t = t >> KSH;
                if (t >  127L) t =  127L;
                if (t < -128L) t = -128L;
                KU[d][cella] = (unsigned char)(t + 128L);
                ksum += (unsigned)KU[d][cella];
                t = (long)vv[d] + 16L;
                t = t >> VSH;
                if (t >  127L) t =  127L;
                if (t < -128L) t = -128L;
                VU[d][cella] = (unsigned char)(t + 128L);
            }
            KSUM[cella] = ksum;
            c_alt += tmr_stop();

            tmr_start();
            attenzione(l, pos);
            cc = tmr_stop();
            c_att += cc;
            c_attp[pos] += cc;
            verifica_vec(pos, 1 + l * 4 + 2, cat, DIM);

            s = quant_shift(cat, DIM);
            tmr_start(); build(64);          c_build += tmr_stop();
            tmr_start(); matvec(m + 3, o, s); c_mac += tmr_stop();

            tmr_start();
            for (i = 0; i < DIM; i++) {
                t = (long)x[i] + (long)o[i];
                if (t >  32767L) t =  32767L;
                if (t < -32768L) t = -32768L;
                x[i] = (int)t;
            }
            c_alt += tmr_stop();

            tmr_start(); rmsnorm(xb, x, NRM_FN[l], DIM); c_alt += tmr_stop();

            s = quant_shift(xb, DIM);
            tmr_start(); build(64);          c_build += tmr_stop();
            tmr_start();
            matvec(m + 4, h1, s);
            matvec(m + 6, h3, s);
            c_mac += tmr_stop();

            tmr_start();
            eflash.bank = BANCOK;
            for (i = 0; i < HID; i++) {
                t = (long)h1[i];
                t = t * 24L;
                t = t >> QB;
                t = t + 128L;
                if (t < 0)   t = 0;
                if (t > 255) t = 255;
                t = (long)SILU[(int)t];
                t = t * (long)h3[i];
                t = t >> QB;
                if (t >  32767L) t =  32767L;
                if (t < -32768L) t = -32768L;
                h1[i] = (int)t;
                if ((i & 63) == 0) tmr_poll();
            }
            c_alt += tmr_stop();

            s = quant_shift(h1, HID);
            tmr_start(); build(176);         c_build += tmr_stop();
            tmr_start(); matvec(m + 5, o, s); c_mac += tmr_stop();

            tmr_start();
            for (i = 0; i < DIM; i++) {
                t = (long)x[i] + (long)o[i];
                if (t >  32767L) t =  32767L;
                if (t < -32768L) t = -32768L;
                x[i] = (int)t;
            }
            c_alt += tmr_stop();
            verifica_vec(pos, 1 + l * 4 + 3, x, DIM);
        }

        /* --- output head: qui esce un TOKEN, non un vettore --- */
        tmr_start();
        rmsnorm(xb, x, NRM_FIN, DIM);
        c_alt += tmr_stop();
        verifica_vec(pos, 1 + LAYERS * 4 + 0, xb, DIM);

        tmr_start();
        i = output_head(&t);
        c_oh += tmr_stop();
        verifica(pos, 1 + LAYERS * 4 + 1, LG0);
        verifica(pos, 1 + LAYERS * 4 + 2, t);
        verifica(pos, 1 + LAYERS * 4 + 3, (long)i);
        if (pos == 0) { R0MAC = LG0; R0RIF = LG0RIF; }
        GEN[pos] = i;
    }

    putnum((long)confronti); put(p" confronti, ");
    putnum((long)errori); put(p" errori"); nl();
    if (errori == 0) {
        put(p"TRANSFORMER COMPLETO OK"); nl();
    } else {
        put(p"primo: p"); putnum((long)pr_pos);
        put(p" st");      putnum((long)pr_st);
        put(p" ");        putnum(pr_ott);
        put(p" != ");     putnum(pr_att); nl();
        put(p"logit0 p0: mac "); putnum(R0MAC);
        put(p" C "); putnum(R0RIF); nl();
    }
    nl();

    /* per posizione = per token, senza output head: il ciclo interno fa
       gia' tutti e quattro i layer */
    c_mac   /= (unsigned long)NPOS;
    c_build /= (unsigned long)NPOS;
    c_alt   /= (unsigned long)NPOS;
    c_att   /= (unsigned long)NPOS;
    c_oh    /= (unsigned long)NPOS;
    tot = c_mac + c_build + c_alt + c_att + c_oh;

    put(p"per token, 4 layer + head:"); nl();
    put(p"matvec : "); putnum((long)c_mac);   nl();
    put(p"build  : "); putnum((long)c_build); nl();
    put(p"norm+  : "); putnum((long)c_alt);
    put(p"  rope silu res"); nl();
    put(p"attenz : "); putnum((long)c_att);
    put(p"  media su "); putnum((long)NPOS); put(p" pos"); nl();
    put(p"head   : "); putnum((long)c_oh); nl();
    put(p"TOTALE : "); putnum((long)tot); nl();
    put(p"         ");
    putnum((long)(tot / 985248UL));
    put(p",");
    putnum((long)((tot % 985248UL) * 10UL / 985248UL));
    put(p" s per parola"); nl();

    put(p"token generati:"); nl();
    for (i = 0; i < NPOS; i++) {
        putnum((long)GEN[i]);
        cwin_put_char(&cw, p' ', 7);
    }
    nl();

    /* La curva vera dell'attenzione per posizione, in migliaia di cicli.
       Non estrapolare da due punti: la forma si guarda. */
    put(p"attenzione per pos, migliaia:"); nl();

    for (i = 0; i < NPOS; i++) {
        putnum((long)(c_attp[i] / 1000UL));
        cwin_put_char(&cw, p' ', 7);
        if ((i & 7) == 7) nl();
    }

    while (true)
        ;
    return 0;
}
'''

sorgente = (sorgente.replace("__REGIONI__", REGIONI)
                    .replace("__BANCHI__", BANCHI_C)
                    .replace("__DESCR__", DESCR)
                    .replace("__SWITCH__", SWITCH))
open("l04.c", "w", encoding="utf-8").write(sorgente)

righe_asm = sum(kernel(banco_bpr[b], "av").count("\n") + 1 for b in BANCHI)
print("l04.c scritto: %d kernel nei banchi, %d righe di assembly in tutto"
      % (len(BANCHI), righe_asm))
print("la compilazione e' lunga: sono %d righe di assembly inline" % righe_asm)
print("COMPILARE CON -O2, NON -O3")
