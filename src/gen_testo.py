"""gen_testo.py - writes gen.c: THE REAL PROGRAM. It generates text.

Not a test bench: no targets, no checks, no timers. It starts from BOS,
generates NGEN tokens one after another reusing each one it just produced,
and prints them.

HOW IT IS VERIFIED. Not by comparisons: **by the text it writes**. It must
produce the reference sentence, the one generate_fx.exe and oracolo_att.exe
give on the PC:

    Once upon a time, there was a little girl named Lily. She loved to
    play outside in the sunshine.

If that comes out, everything underneath is right - a stricter check than
the 336 comparisons in l04.c, because here every token depends on all the
previous ones instead of on supplied values.

RELATION TO l04.c. l04.c stays the test bench with its 336 comparisons;
gen.c is the program. The shared parts - bank plan, ternary kernel, 16x8
MAC, scales - are IMPORTED from gen_l04.py, so they cannot drift apart.
What is written here is only what l04.c does not have: the input embedding
read from the banks, the generation loop, the vocabulary and the printing.

THREE SUBSTANTIVE DIFFERENCES from l04.c:

1. SEQ 48 instead of 64. The RAM budget does not survive 64 with the output
   head: the cache and the accumulators grow with SEQ and ~2.5 KB would be
   missing. 48 is enough for a 35-token sentence. tables.bin is NOT touched:
   the RoPE tables stay at 64 positions and only the first 48 are used, so
   the values are the same as always and the oracle stays comparable.

2. The input embedding is read from the TRANSPOSED banks, the same ones the
   output head uses: emb[token][d] lives at embT[d][token]. That is 128
   scattered reads instead of 128 contiguous ones, but it saves four banks
   and above all keeps ONE SINGLE COPY of the model: two copies that can
   fall out of step are exactly what has already cost time here.

3. The output head works on half the vocabulary at a time. The accumulators
   are 256 instead of 512 - 1536 bytes saved - and it costs NOTHING: for
   each dimension half a row is copied instead of the whole row, so the
   number of bytes copied is the same.

Usage:  python gen_testo.py
        .\\c64.ps1 gen -Cicli 1500000000

Note: inline comments below are in Italian.
"""

import re
import struct
import sys

import gen_l04 as L      # ⚠️ rigenera anche l04.c: e' voluto, restano allineati

D = L.D
LAYERS = L.LAYERS
KVD = L.KVDIM
HID = L.HIDDEN
VOC = L.VOCAB
BANCO = L.BANCO
DPB = L.DPB
BANCOK = L.BANCOK
BANCOM = L.BANCOM
BANCOE = L.BANCOE

SEQG = 48                 # posizioni di contesto: 4*48 = 192, sta in un byte
# ⚠️ 34, non 35.  L'array di riferimento dell'oracolo ha 35 elementi ma il
# primo e' il BOS, che non e' generato: i token prodotti sono 34, e il 34esimo
# e' il punto che chiude "...in the sunshine."  Con 35 esce anche ' One',
# l'inizio della frase dopo, e sembra che il testo sia troncato.
NGEN = 34

# Token di innesco.  0 = la frase di riferimento (solo BOS).  Con un valore
# diverso si parte da BOS + quel token, esattamente come oracolo_att.exe
# -seq N, e si confrontano gli id generati.
# ⚠️ SERVE.  Il programma intero e' stato provato su UN solo ingresso, e per
# giunta quello su cui e' stata messa a punto tutta la pipeline: sec. 6 dice
# che non e' un campione.  Le 150 prove di fedelta' erano sulla SIMULAZIONE
# della forma C64, non sul binario.
PROVA_TOK = ""
if "--tok" in sys.argv:
    PROVA_TOK = sys.argv[sys.argv.index("--tok") + 1]
    sys.argv = [a for a in sys.argv if a != "--tok" and a != PROVA_TOK]
SEME = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NP = 2 if SEME else 1     # quanti token sono prompt e non si sovrascrivono
if SEME:
    print("innesco: BOS + token %d (confrontare con oracolo_att.exe -seq %d)"
          % (SEME, SEME))
CHUNK = VOC // 2          # meta' vocabolario per volta nell'output head
NT2 = LAYERS * SEQG
assert NT2 <= 256, "l*SEQ+t deve stare in un byte: e' l'indice nel registro X"
assert NGEN <= SEQG

print("SEQ %d, %d token, %d celle di cache, output head a meta' (%d)"
      % (SEQG, NGEN, NT2, CHUNK))

# ------------------------------------------------------------ RoPE fino a NGEN
t = open("tables.bin", "rb").read()
base_cos = (256 + 512 + 256) * 2
base_sin = base_cos + 64 * 8 * 2      # la tabella su disco ha 64 posizioni
cos = [struct.unpack_from("<8h", t, base_cos + p * 16) for p in range(SEQG)]
sin = [struct.unpack_from("<8h", t, base_sin + p * 16) for p in range(SEQG)]

# ------------------------------------------------------------- vocabolario
# Convertito a PETSCII qui, una volta, invece che sul C64 a ogni carattere.
# Le minuscole ASCII diventano i codici delle maiuscole, che nel set di
# caratteri di partenza sono le lettere che si vedono; il resto dello spazio
# stampabile ASCII coincide col PETSCII.
righe = open("vocab.txt", "rb").read().split(b"\n")
righe = [s.rstrip(b"\r") for s in righe][:VOC]
assert len(righe) == VOC, "vocab.txt ha %d voci su %d" % (len(righe), VOC)


def petscii(s):
    """ASCII -> PETSCII per il set di caratteri MISTO (vic_setmode a $1800).

    ⚠️ Li' i codici $41-$5A sono le MINUSCOLE e le maiuscole stanno a
    $C1-$DA.  Passando le maiuscole ASCII invariate escono minuscole: la
    frase veniva 'once upon a time' invece di 'Once upon a time'."""
    fuori = []
    for c in s:
        if 97 <= c <= 122:
            fuori.append(c - 32)          # minuscole ASCII -> $41-$5A
        elif 65 <= c <= 90:
            fuori.append(c + 128)         # MAIUSCOLE ASCII -> $C1-$DA
        elif 32 <= c <= 64:
            fuori.append(c)               # spazio, punteggiatura, cifre
        else:
            fuori.append(63)              # tutto il resto: punto interrogativo
    return fuori


testo, offset, lung = [], [], []
for s in righe:
    offset.append(len(testo))
    p = petscii(s)
    lung.append(len(p))
    testo += p
print("vocabolario: %d byte di testo + %d di indice"
      % (len(testo), 2 * VOC + VOC))
print("  token 403,407,261 -> %s" % [bytes(petscii(righe[i])).decode("latin1")
                                     for i in (403, 407, 261)])


def vet(v):
    return ",".join(str(x) for x in v)


# --------------------------------------------------------------- gencost.h
with open("gencost.h", "w", encoding="utf-8") as g:
    g.write("/* generato da gen_testo.py - dentro #pragma data( bdatak ) */\n")
    g.write("const int SILU[256] = {%s};\n" % vet(L.tsilu))
    g.write("const int TEXP[512] = {%s};\n" % vet(L.texp))
    g.write("const int RSQ[256] = {%s};\n" % vet(L.trsq))
    g.write("const int RCOS[%d][8] = {%s};\n"
            % (SEQG, ",".join("{%s}" % vet(c) for c in cos)))
    g.write("const int RSIN[%d][8] = {%s};\n"
            % (SEQG, ",".join("{%s}" % vet(s) for s in sin)))
    g.write("const int NRM_AN[%d][%d] = {%s};\n"
            % (LAYERS, D, ",".join("{%s}" % vet(v) for v in L.NRM_AN)))
    g.write("const int NRM_FN[%d][%d] = {%s};\n"
            % (LAYERS, D, ",".join("{%s}" % vet(v) for v in L.NRM_FN)))
    g.write("const int NRM_FIN[%d] = {%s};\n" % (D, vet(L.NRM_FINAL)))
    g.write("const unsigned ESUM[%d] = {%s};\n" % (VOC, vet(L.ESUM)))
    g.write("const unsigned VOFF[%d] = {%s};\n" % (VOC, vet(offset)))
    g.write("const unsigned char VLEN[%d] = {%s};\n" % (VOC, vet(lung)))
    g.write("const unsigned char VTXT[%d] = {%s};\n" % (len(testo), vet(testo)))
print("gencost.h scritto")

# ------------------------------------------------------- il MAC, terza forma
# macoh qui fa 256 token per volta invece di 512: un anello solo, e gli
# accumulatori sono la meta'.
asm_o = L.puntatori("PTRO", 0)
asm_o += ["        ldx #0", "lpo:", "        lda ROW,x", "        tay"]
asm_o += L.prodotto(L.PZ_PTR, L.PZ_PTR + 4, "OHH0", "OHH1", "OHH2", "nho", True)
asm_o += L.prodotto(L.PZ_PTR + 8, L.PZ_PTR + 12, "OHL0", "OHL1", "OHL2",
                    "nlo", True)
asm_o += ["        inx", "        bne lpo"]

asm_k = []
for d in range(KVD):
    asm_k.append("        // ---- punteggi, d = %d ----" % d)
    asm_k += L.puntatori("PTRK", d)
    asm_k += ["        ldx XBASE", "lpk%d:" % d,
              "        lda KU + %d,x" % (d * NT2), "        tay"]
    asm_k += L.prodotto(L.PZ_PTR, L.PZ_PTR + 4, "ACH0", "ACH1", "ACH2",
                        "nhk%d" % d, True)
    asm_k += L.prodotto(L.PZ_PTR + 8, L.PZ_PTR + 12, "ACL0", "ACL1", "ACL2",
                        "nlk%d" % d, True)
    asm_k += ["        inx", "        cpx NTC", "        bne lpk%d" % d]

asm_v = ["        ldx XCUR"] + L.puntatori("PTRV", 0)
for d in range(KVD):
    asm_v.append("        // ---- media pesata, d = %d ----" % d)
    asm_v.append("        ldy VU + %d,x" % (d * NT2))
    asm_v += L.prodotto(L.PZ_PTR, L.PZ_PTR + 4, "VH0 + %d" % d,
                        "VH1 + %d" % d, "VH2 + %d" % d, "nhv%d" % d, False)
    asm_v += L.prodotto(L.PZ_PTR + 8, L.PZ_PTR + 12, "VL0 + %d" % d,
                        "VL1 + %d" % d, "VL2 + %d" % d, "nlv%d" % d, False)

# ---------------------------------------------------- sezioni dei banchi
pezzi = []
for banco in L.BANCHI:
    dentro = sorted(i for i in L.piano if L.piano[i][0] == banco)
    pezzi.append("#pragma code ( bcode%d )" % banco)
    pezzi.append("#pragma data ( bdata%d )" % banco)
    for i in dentro:
        _, nbyte, _, fn = L.piano[i]
        pezzi.append("__export const char %s[%d] = {" % (L.NOMI[i], nbyte))
        pezzi.append("#embed %d 0 \"%s\"" % (nbyte, fn))
        pezzi.append("};")
    pezzi.append("__noinline void macrow_b%d(void)" % banco)
    pezzi.append("{\n    __asm\n    {")
    pezzi.append(L.kernel(L.banco_bpr[banco], "av_b%d" % banco))
    pezzi.append("    }\n}")

pezzi.append("/* ---- banco %d: MAC 16x8 e tabelle dei quarti di quadrato ---- */"
             % BANCOM)
pezzi.append("#pragma code ( bcode%d )" % BANCOM)
pezzi.append("#pragma data ( bdata%d )" % BANCOM)
for nome, tab, alto in (("T1L", L.QS1, False), ("T1H", L.QS1, True),
                        ("T2L", L.QS2, False), ("T2H", L.QS2, True)):
    pezzi.append("__export const unsigned char %s[512] = {%s};"
                 % (nome, L.bytes_di(tab, alto)))
    pezzi.append("#pragma align( %s, 256 )" % nome)
for nome, corpo in (("macq_k", asm_k), ("macq_v", asm_v), ("macoh", asm_o)):
    pezzi.append("__noinline void %s(void)\n{\n    __asm\n    {" % nome)
    pezzi.append("\n".join(corpo))
    pezzi.append("    }\n}")

pezzi.append("/* ---- banchi %d-%d: embedding trasposto ---- */"
             % (BANCOE, BANCOE + 3))
for i in range(4):
    pezzi.append("#pragma data ( bdataE%d )" % i)
    pezzi.append("__export const char EMBT%d[%d] = {" % (i, BANCO))
    pezzi.append("#embed %d 0 \"embT%d.bin\"" % (BANCO, i))
    pezzi.append("};")

pezzi.append("/* ---- banco %d: costanti e vocabolario ---- */" % BANCOK)
pezzi.append("#pragma data ( bdatak )")
pezzi.append("#include \"gencost.h\"")
pezzi.append("#pragma code ( code )")
pezzi.append("#pragma data ( data )")
BANCHI_C = "\n".join(pezzi)

REGIONI = "\n".join(
    "#pragma section( bcode%d, 0 )\n#pragma section( bdata%d, 0 )\n"
    "#pragma region( bank%d, 0x8000, 0xc000, , %d, { bcode%d, bdata%d } )"
    % (b, b, b, b, b, b) for b in L.BANCHI + [BANCOM])
REGIONI += ("\n#pragma section( bdatak, 0 )\n"
            "#pragma region( bankk, 0x8000, 0xc000, , %d, { bdatak } )" % BANCOK)
for i in range(4):
    REGIONI += ("\n#pragma section( bdataE%d, 0 )\n"
                "#pragma region( bankE%d, 0x8000, 0xc000, , %d, { bdataE%d } )"
                % (i, i, BANCOE + i, i))

RIGHE = [L.piano[i][1] // L.piano[i][2] for i in range(LAYERS * 7)]
DESCR = ("const char * const MPTR[%d] = {%s};\n"
         % (LAYERS * 7, ",".join(L.NOMI[i] for i in range(LAYERS * 7))) +
         "const unsigned char MBANCO[%d] = {%s};\n"
         % (LAYERS * 7, ",".join(str(L.piano[i][0]) for i in range(LAYERS * 7))) +
         "const int MRIGHE[%d] = {%s};\n" % (LAYERS * 7, vet(RIGHE)) +
         "const unsigned MANTM[%d] = {%s};\n" % (len(L.MANT), vet(L.MANT)) +
         "const unsigned char SHIFTM[%d] = {%s};\n"
         % (len(L.SHIFT), vet(L.SHIFT)) +
         "const char * const EMBP[4] = { EMBT0, EMBT1, EMBT2, EMBT3 };")

SWITCH = "\n".join("        case %d:  macrow_b%d(); break;" % (b, b)
                   for b in L.BANCHI[:-1])
SWITCH += "\n        default: macrow_b%d(); break;" % L.BANCHI[-1]

# ------------------------------------------------------------------- gen.c
sorgente = r'''/* gen.c - GENERATO DA gen_testo.py.  IL PROGRAMMA VERO: genera testo.

   Parte da BOS e produce NGEN token, riusando ogni volta quello appena
   generato.  Nessun bersaglio, nessuna verifica, nessun cronometro.

   ⚠️ SI VERIFICA COL TESTO CHE SCRIVE.  Deve uscire la frase che danno
   generate_fx.exe e oracolo_att.exe sul PC.  E' un controllo piu' severo
   dei 336 confronti di l04.c, perche' qui ogni token dipende da tutti i
   precedenti.

   ⚠️ COMPILARE CON -O2, NON -O3 (sec. 11).
   Eseguire:  .\c64.ps1 gen -Cicli 1500000000
*/

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include <c64/keyboard.h>
#include "gendata.h"

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

unsigned char PL[2816];
unsigned char PH[2816];
int rowres;

/* cache trasposta e spostata di +128, cella = l*SEQG + t (sec. 9-ter) */
unsigned char KU[KVD][NT2];
unsigned char VU[KVD][NT2];
unsigned      KSUM[NT2];
unsigned char ACH0[NT2], ACH1[NT2], ACH2[NT2];
unsigned char ACL0[NT2], ACL1[NT2], ACL2[NT2];
unsigned char VH0[KVD], VH1[KVD], VH2[KVD];
unsigned char VL0[KVD], VL1[KVD], VL2[KVD];
unsigned      PTRK[KVD][8];
unsigned      PTRV[8];
unsigned char XBASE, NTC, XCUR;

/* output head a meta' vocabolario per volta: accumulatori da CHUNK e non
   da VOC, e si copia meta' riga per volta, quindi non costa niente */
unsigned char ROW[CHUNK];
unsigned char OHH0[CHUNK], OHH1[CHUNK], OHH2[CHUNK];
unsigned char OHL0[CHUNK], OHL1[CHUNK], OHL2[CHUNK];
unsigned      PTRO[8];

__BANCHI__

__DESCR__

int x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], cat[DIM], o[DIM];
int h1[HID], h3[HID];
int xq[HID];
long att[SEQG];
int  ids[SEQG];
int  GEN[NGEN];      /* gli id generati, per confrontarli con l'oracolo */

static CharWin cw;

static void put(const char * s) { cwin_put_string(&cw, s, VCOL_LT_BLUE); }
static void nl(void)            { cwin_cursor_newline(&cw); }

static void putnum(long v)
{
    char buf[12];
    char i = 12;
    unsigned long u;
    if (v < 0) { cwin_put_char(&cw, p'-', VCOL_LT_BLUE); u = (unsigned long)(-v); }
    else       u = (unsigned long)v;
    if (u == 0) { cwin_put_char(&cw, p'0', VCOL_LT_BLUE); return; }
    while (u) { buf[--i] = p'0' + (char)(u % 10); u /= 10; }
    while (i < 12) cwin_put_char(&cw, buf[i++], VCOL_LT_BLUE);
}

/* ---------------------------------------------------------------- INPUT
   Lettura da tastiera senza KERNAL (keyb_poll scansiona la matrice) e
   tokenizzatore greedy longest-match sullo stesso vocabolario che serve
   alla stampa.  Il confronto e' PETSCII contro PETSCII: VTXT e' gia'
   convertito, e la tastiera restituisce gli stessi codici.
   ⚠️ Distingue maiuscole e minuscole, come il tokenizzatore vero: " One"
   e " one" sono token diversi, quindi lo shift conta. */
static char inbuf[IN_MAX];
static char inlen = 0;

static void leggi_riga(void)
{
    char k, prec = 0;
    put(p"> ");
    for (;;) {
        keyb_poll();
        k = (keyb_key & KSCAN_QUAL_DOWN) ? keyb_codes[keyb_key & 0x7f] : 0;
        if (k == prec) continue;          /* niente ripetizione automatica */
        prec = k;
        if (k == 0) continue;
        if (k == KEY_RETURN) { nl(); return; }
        if (k == KEY_DEL) {
            if (inlen > 0) {
                inlen--;
                cwin_cursor_left(&cw);
                cwin_put_char(&cw, p' ', VCOL_LT_BLUE);
                cwin_cursor_left(&cw);
            }
            continue;
        }
        if (inlen < IN_MAX - 1) {
            inbuf[inlen++] = k;
            cwin_put_char(&cw, k, VCOL_LT_BLUE);
        }
    }
}

/* Il pezzo piu' lungo del vocabolario che combacia a partire da pos.
   Si scarta subito su primo byte diverso: 512 voci per posizione sono
   poche decine di migliaia di cicli, non si nota. */
static int piu_lungo(char pos, char * quanti)
{
    int t, mig = -1;
    char i, n, migl = 0;
    unsigned o;
    eflash.bank = BANCOK;
    for (t = 259; t < VOC; t++) {
        n = VLEN[t];
        if (n == 0 || n <= migl || pos + n > inlen) continue;
        o = VOFF[t];
        for (i = 0; i < n; i++)
            if (VTXT[o + i] != inbuf[pos + i]) break;
        if (i == n) { mig = t; migl = n; }
    }
    *quanti = migl;
    return mig;
}

static char tokenizza(void)
{
    char pos = 0, n, quanti;
    int t;
    n = 1;                                /* ids[0] e' il BOS */
    while (pos < inlen && n < SEQG - 4) {
        t = piu_lungo(pos, &quanti);
        if (t < 0) { pos++; continue; }   /* carattere non rappresentabile */
        ids[n++] = t;
        pos += quanti;
    }
    return n;
}

/* la stringa del token, gia' in PETSCII dentro il banco delle costanti */
static void stampa_token(int tok)
{
    unsigned o;
    unsigned char n, i;
    eflash.bank = BANCOK;
    o = VOFF[tok];
    n = VLEN[tok];
    for (i = 0; i < n; i++) cwin_put_char(&cw, VTXT[o + i], VCOL_LT_BLUE);
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

/* ⚠️ Lo shift di ampiezza variabile (sh = 9-15) e' un ciclo di un giro per
   bit.  La parte a byte interi si stacca - per lo shift aritmetico
   (t>>8)>>k == t>>(8+k) - e resta un residuo di al piu' 7 bit: -696k cicli
   per token, esatti (sec. 7). */
static void matvec(int m, int * y, char sx)
{
    int i, righe;
    long t, mant;
    char banco = MBANCO[m];
    char sh = SHIFTM[m] - sx;
    char pre8 = 0;

    if (sh >= 8) { pre8 = 1; sh = sh - 8; }

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
    }
}

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

static void attenzione(int l, int pos)
{
    int h, d, t;
    long s, best, somma, inv;
    long sq, sp, ach, acl;
    int qq;
    unsigned char qhu, ql, base, xc;
    unsigned pw;

    base = (unsigned char)(l * SEQG);

    for (h = 0; h < HEADS; h++) {
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
            xc = base + (unsigned char)t;
            ACH0[xc] = 0; ACH1[xc] = 0; ACH2[xc] = 0;
            ACL0[xc] = 0; ACL1[xc] = 0; ACL2[xc] = 0;
        }
        XBASE = base;
        NTC   = base + (unsigned char)pos + 1;
        eflash.bank = BANCOM;
        macq_k();

        eflash.bank = BANCOK;
        best = -2147483647L;
        for (t = 0; t <= pos; t++) {
            xc = base + (unsigned char)t;
            ach = (long)ACH0[xc] + ((long)ACH1[xc] << 8) + ((long)ACH2[xc] << 16);
            acl = (long)ACL0[xc] + ((long)ACL1[xc] << 8) + ((long)ACL2[xc] << 16);
            ach = ach - 128L * (long)KSUM[xc];
            ach = ach << 8;
            ach = ach + acl;
            ach = ach - sq;
            ach = ach >> (QB + 2 - KSH);
            att[t] = ach;
            if (ach > best) best = ach;
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
            qhu = (unsigned char)(pw >> 8);
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
        }
        sp = sp << 7;

        eflash.bank = BANCOK;
        for (d = 0; d < HDIM; d++) {
            ach = (long)VH0[d] + ((long)VH1[d] << 8) + ((long)VH2[d] << 16);
            acl = (long)VL0[d] + ((long)VL1[d] << 8) + ((long)VL2[d] << 16);
            ach = ach << 8;
            ach = ach + acl;
            ach = ach - sp;
            ach = ach >> (15 - VSH);
            if (ach >  32767L) ach =  32767L;
            if (ach < -32768L) ach = -32768L;
            cat[h * HDIM + d] = (int)ach;
        }
    }
}

/* Output head, meta' vocabolario per volta.  Restituisce l'argmax. */
static int output_head(void)
{
    int d, t, best, blocco;
    long sxb, ach, acl, bestv;
    unsigned char xbhu, xbl;
    const char * src;
    int qq;

    sxb = 0;
    for (d = 0; d < DIM; d++) sxb += (long)xb[d];
    sxb = sxb << 7;

    best  = 0;
    bestv = -2147483647L;

    for (blocco = 0; blocco < 2; blocco++) {
        for (t = 0; t < CHUNK; t++) {
            OHH0[t] = 0; OHH1[t] = 0; OHH2[t] = 0;
            OHL0[t] = 0; OHL1[t] = 0; OHL2[t] = 0;
        }
        for (d = 0; d < DIM; d++) {
            eflash.bank = BANCOE + (d / DPB);
            src = EMBP[d / DPB] + (long)(d % DPB) * VOC + blocco * CHUNK;
            for (t = 0; t < CHUNK; t++) ROW[t] = (unsigned char)src[t];

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
        }

        eflash.bank = BANCOK;
        for (t = 0; t < CHUNK; t++) {
            ach = (long)OHH0[t] + ((long)OHH1[t] << 8) + ((long)OHH2[t] << 16);
            acl = (long)OHL0[t] + ((long)OHL1[t] << 8) + ((long)OHL2[t] << 16);
            ach = ach - 128L * (long)ESUM[blocco * CHUNK + t];
            ach = ach << 8;
            ach = ach + acl;
            ach = ach - sxb;
            if (ach > bestv) { bestv = ach; best = blocco * CHUNK + t; }
        }
    }
    return best;
}

int main(void)
{
    int i, l, m, pos, d, b, tok;
    char np;
    char s;
    long t;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);

    /* ⚠️ In cartuccia il KERNAL non ha fatto la sua init, quindi i registri
       di colore del VIC restano come capita: schermo nero.  I valori di
       accensione di un C64 vanno rimessi a mano, se no lo schermo non e'
       quello che la gente riconosce. */
    vic.color_border = VCOL_LT_BLUE;
    vic.color_back   = VCOL_BLUE;

    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);
    cwin_fill(&cw, 32, VCOL_LT_BLUE);   /* anche la RAM colore, non solo i caratteri */

    /* Schermata di titolo.  Una cartuccia prende il controllo prima che la
       ROM del BASIC stampi il suo banner, quindi quello non compare: e' come
       funzionavano tutte le cartucce dal 1982 in poi.  Quello che facevano,
       pero', era avere un titolo proprio, ed e' questo.
       ⚠️ Niente "BASIC V2" e niente "BYTES FREE": sarebbero affermazioni
       false, perche' il BASIC non c'e' e quella memoria non e' libera. */
    put(p"       **** COMMODORE 64 ****"); nl(); nl();
    put(p"   TERNARY.C64   "); putnum(688128L); put(p" WEIGHTS"); nl();
    put(p"   MODEL IN CARTRIDGE, 22 BANKS"); nl(); nl();

    ids[0] = 1;                       /* BOS */

    /* Il prompt e' facoltativo: RETURN a vuoto e parte da solo, come prima.
       ⚠️ Ogni token di prompt costa un passaggio completo del modello, cioe'
       36 secondi: non e' gratis come su una macchina moderna. */
#ifdef PROVA_TOK
    /* banco di prova: stessa strada del codice vero, senza la tastiera */
    for (i = 0; i < PROVA_LEN; i++) inbuf[i] = PROVA[i];
    inlen = PROVA_LEN;
    np = tokenizza();
    put(p"tok:"); nl();
    for (i = 1; i < np; i++) { putnum((long)ids[i]); cwin_put_char(&cw, p' ', VCOL_LT_BLUE); }
    nl();
    for (i = 1; i < np; i++) stampa_token(ids[i]);
    nl(); nl(); put(p"end of test");
    while (true) ;
#endif
    put(p"type a beginning, or press return"); nl();
    leggi_riga();
    np = tokenizza();
    if (np > 1) {
        put(p"prompt: ");
        for (i = 1; i < np; i++) stampa_token(ids[i]);
        nl();
    }
    nl();

    for (pos = 0; pos < NGEN; pos++) {
        tok = ids[pos];

        /* embedding di ingresso, dai banchi TRASPOSTI: emb[tok][d] sta a
           embT[d][tok].  Quattro commutazioni, non centoventotto. */
        for (b = 0; b < 4; b++) {
            eflash.bank = BANCOE + b;
            for (i = 0; i < DPB; i++) {
                d = b * DPB + i;
                t = (long)((int)(unsigned char)EMBP[b][(long)i * VOC + tok]
                           - 128);
                t = t * EMB_MANT;
                t = t >> (EMB_SHIFT - QB);
                if (t >  32767L) t =  32767L;
                if (t < -32768L) t = -32768L;
                x[d] = (int)t;
            }
        }

        for (l = 0; l < LAYERS; l++) {
            m = l * 7;

            rmsnorm(xb, x, NRM_AN[l], DIM);
            s = quant_shift(xb, DIM);
            build(64);
            matvec(m + 0, q,  s);
            matvec(m + 1, kk, s);
            matvec(m + 2, vv, s);

            for (i = 0; i < HEADS; i++) rope(q + i * HDIM, pos);
            rope(kk, pos);

            /* cache: int8 ARROTONDANDO, trasposta, spostata di +128 */
            {
                unsigned char cella = (unsigned char)(l * SEQG + pos);
                unsigned ksum = 0;
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
            }

            attenzione(l, pos);

            s = quant_shift(cat, DIM);
            build(64);
            matvec(m + 3, o, s);
            for (i = 0; i < DIM; i++) {
                t = (long)x[i] + (long)o[i];
                if (t >  32767L) t =  32767L;
                if (t < -32768L) t = -32768L;
                x[i] = (int)t;
            }

            rmsnorm(xb, x, NRM_FN[l], DIM);
            s = quant_shift(xb, DIM);
            build(64);
            matvec(m + 4, h1, s);
            matvec(m + 6, h3, s);

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
            }

            s = quant_shift(h1, HID);
            build(176);
            matvec(m + 5, o, s);
            for (i = 0; i < DIM; i++) {
                t = (long)x[i] + (long)o[i];
                if (t >  32767L) t =  32767L;
                if (t < -32768L) t = -32768L;
                x[i] = (int)t;
            }
        }

        rmsnorm(xb, x, NRM_FIN, DIM);
        tok = output_head();
        /* i token di prompt non si sovrascrivono */
        if (pos + 1 >= np && pos + 1 < SEQG) ids[pos + 1] = tok;
        GEN[pos] = tok;

        /* la parola compare appena e' pronta: mezzo minuto di attesa e poi
           una parola, che e' esattamente la cosa da far vedere */
        if (pos + 1 >= np) stampa_token(tok);
    }

    /* Gli id NON si stampano piu': servivano a confrontare le sequenze con
       l'oracolo, e in un filmato sono quattro righe di numeri senza
       spiegazione sotto una frase in inglese.  Il confronto con l'oracolo
       resta possibile da l04, che e' il posto giusto per i numeri. */
    nl(); nl();
    put(p"done.");

    while (true)
        ;
    return 0;
}
'''

sorgente = (sorgente.replace("__REGIONI__", REGIONI)
                    .replace("__BANCHI__", BANCHI_C)
                    .replace("__DESCR__", DESCR)
                    .replace("__SWITCH__", SWITCH))
open("gen.c", "w", encoding="utf-8").write(sorgente)

with open("gendata.h", "w", encoding="utf-8") as g:
    g.write("/* generato da gen_testo.py */\n")
    g.write("#define SEQG %d\n" % SEQG)
    g.write("#define NGEN %d\n" % NGEN)
    g.write("#define SEME %d\n" % SEME)
    g.write("#define NP %d\n" % NP)
    g.write("#define NT2 %d\n" % NT2)
    g.write("#define VOC %d\n" % VOC)
    g.write("#define CHUNK %d\n" % CHUNK)
    g.write("#define BANCOK %d\n" % BANCOK)
    g.write("#define BANCOM %d\n" % BANCOM)
    g.write("#define BANCOE %d\n" % BANCOE)
    g.write("#define DPB %d\n" % DPB)
    g.write("#define IN_MAX 32   /* caratteri digitabili nel prompt */\n")
    if PROVA_TOK:
        # banco di prova del tokenizzatore: niente tastiera, testo fisso.
        # Serve a esercitare piu_lungo() e tokenizza() sul C64 e confrontare
        # gli id con la versione Python, senza dover digitare in VICE.
        pt = petscii(PROVA_TOK.encode("latin1"))
        g.write("#define PROVA_TOK 1\n")
        g.write("#define PROVA_LEN %d\n" % len(pt))
        g.write("const char PROVA[%d] = {%s};\n" % (len(pt), vet(pt)))
    g.write("#define EMB_MANT %d\n" % L.EMB_MANT)
    g.write("#define EMB_SHIFT %d\n" % L.EMB_SHIFT)

print("gen.c e gendata.h scritti")
print("COMPILARE CON -O2, NON -O3")
