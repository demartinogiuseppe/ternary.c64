"""gen_l0full.py — genera l0full.c: LAYER 0 COMPLETO sul C64.

STORIA DEI BUG SU QUESTO FILE

1) Offset sbagliati nell'estrazione delle RMSNorm -> numeri astronomici.
   Risolto leggendo da scale.h, prodotto da export_banchi.py e verificato.

2) Shift dentro un'espressione composta non applicato: valori 4096 volte
   troppo grandi. Risolto spezzando ogni shift a 32 bit.

3) BUG VERO, trovato leggendo l0full.asm. Il codice generato per il ciclo
   dell'embedding era:

       LDA $1b00,x    ; carica EMB1[i], un byte: -14 cioe' $F2
       STA ACCU + 0   ; lo mette nel byte basso dell'accumulatore

   e basta. NESSUNA ESTENSIONE DEL SEGNO. I tre byte alti restano a zero,
   quindi ACCU vale $000000F2 = 242 invece di -14.

   oscar64 non estende il segno quando promuove un signed char a long.
   Ogni lettura di un signed char destinata all'aritmetica a 32 bit va
   corretta a mano:

       e8 = EMB1[i];
       if (e8 > 127) e8 -= 256;

   Riguarda EMB1[] e xq[] (usato in build()).

Regole per oscar64 con aritmetica mista 16/32 bit:
  - spezzare ogni shift a 32 bit in un passaggio separato
  - non passare long alle funzioni
  - estendere il segno a mano quando si legge un signed char
  - quando qualcosa non torna: leggere il .asm generato, non ipotizzare

Prerequisito: aver lanciato export_banchi.py almeno una volta (crea scale.h).
Bersagli: x0 1155  xb 11251  q -1116  k 757  v 241  o -774
          xb2 4250  sw 10395  o2 7660  x2 8041
"""

import struct
import re
import os

D, KV, H = 128, 16, 352

assert os.path.exists("scale.h"), \
    "scale.h mancante: lancia prima 'python export_banchi.py'"

W = open("model_ternary.bin", "rb").read()

FORME = [("wq", D, D), ("wk", KV, D), ("wv", KV, D), ("wo", D, D),
         ("w1", H, D), ("w2", D, H), ("w3", H, D)]

off = 4 + 7 * 4
scale = {}
for nome, righe, col in FORME:
    scale[nome] = struct.unpack_from("<f", W, off)[0]
    off += 4
    n = righe * col // 4
    open(f"l0_{nome}.bin", "wb").write(W[off:off + n])
    off += n

sh_txt = open("scale.h").read()

def leggi(nome):
    m = re.search(rf"const int {nome}\[128\] = \{{([^}}]+)\}}", sh_txt)
    assert m, f"{nome} non trovata in scale.h"
    return [int(v) for v in m.group(1).split(",")]

an_q = leggi("NRM_L0_an")
fn_q = leggi("NRM_L0_fn")
assert 2000 < max(an_q) < 8192, f"NRM_AN fuori range: max {max(an_q)}"

es_m = int(re.search(r"#define EMB_MANT (\d+)", sh_txt).group(1))
es_s = int(re.search(r"#define EMB_SHIFT (\d+)", sh_txt).group(1))

emb = W[len(W) - 512 * D:]
# EMB1 come int, non signed char: cosi' non serve estensione di segno
emb1 = [b - 256 if b > 127 else b for b in emb[D:2 * D]]


def ms(s):
    e = 0
    while s < 0.5:
        s *= 2.0
        e += 1
    return int(round(s * 32768.0)), 15 + e


t = open("tables.bin", "rb").read()
silu = struct.unpack_from("<256h", t, 0)
rsq  = struct.unpack_from("<256h", t, (256 + 512) * 2)

with open("l0fdata.h", "w") as g:
    g.write("/* generato da gen_l0full.py */\n")
    for nome in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        m, sh = ms(scale[nome])
        g.write(f"#define {nome.upper()}_MANT {m}L\n")
        g.write(f"#define {nome.upper()}_SHIFT {sh}\n")
    g.write(f"#define EMB_MANT {es_m}L\n#define EMB_SHIFT {es_s}\n\n")
    g.write("const int NRM_AN[128] = {" + ",".join(map(str, an_q)) + "};\n")
    g.write("const int NRM_FN[128] = {" + ",".join(map(str, fn_q)) + "};\n")
    # int, non signed char: evita del tutto il problema del segno
    g.write("const int EMB1[128] = {" + ",".join(map(str, emb1)) + "};\n")
    g.write("const int RSQ[256] = {" + ",".join(map(str, rsq)) + "};\n")
    g.write("const int SILU[256] = {" + ",".join(map(str, silu)) + "};\n")

print("l0fdata.h e 7 file .bin scritti")
print("EMB1 dichiarato come int[], non signed char[]")
print("bersagli: x0 1155  xb 11251  q -1116  k 757  v 241  o -774")
print("          xb2 4250  sw 10395  o2 7660  x2 8041")

sorgente = '''/* l0full.c - GENERATO DA gen_l0full.py. Layer 0 completo sul C64.

   Quattro accorgimenti obbligatori su oscar64:
     1. ogni shift a 32 bit spezzato in un passaggio separato su una long
     2. nessun parametro long passato a funzioni: clamp scritto inline
     3. NIENTE signed char nell'aritmetica: oscar64 non estende il segno
        quando promuove a long. EMB1 e xq sono int.
     4. NESSUNA globale che si chiami W1..W4, WW, L1, E0, W16, W32, LB1,
        LS0, LS1, WS1, WB1, zero.  Sono le etichette interne delle routine
        __asm del runtime (mul32, mul32by8, divmod32).  Una globale con
        quel nome le sovrascrive: il salto finisce sull'array, non arriva,
        e l'assemblatore emette spostamento ZERO senza segnalare nulla.
        La moltiplicazione a 32 bit restituisce spazzatura in silenzio.
        Per questo le matrici si chiamano MAT_WQ ... MAT_W3.

   Compilare:  oscar64 -O2 -tf=crt -o=l0full.crt l0full.c
   (Detach in VICE PRIMA di ricompilare)

   Bersagli: x0 1155  xb 11251  q -1116  k 757  v 241  o -774
             xb2 4250  sw 10395  o2 7660  x2 8041 */

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include "l0fdata.h"

#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, heap, stack } )

#pragma section( bdata1, 0 )
#pragma region( bank1, 0x8000, 0xc000, , 1, { bdata1 } )
#pragma data(bdata1)
__export const char MAT_WQ[4096] = {
#embed 4096 0 "l0_wq.bin"
};
__export const char MAT_WK[512] = {
#embed 512 0 "l0_wk.bin"
};
__export const char MAT_WV[512] = {
#embed 512 0 "l0_wv.bin"
};
__export const char MAT_WO[4096] = {
#embed 4096 0 "l0_wo.bin"
};
#pragma data(data)

#pragma section( bdata2, 0 )
#pragma region( bank2, 0x8000, 0xc000, , 2, { bdata2 } )
#pragma data(bdata2)
__export const char MAT_W1[11264] = {
#embed 11264 0 "l0_w1.bin"
};
#pragma data(data)

#pragma section( bdata3, 0 )
#pragma region( bank3, 0x8000, 0xc000, , 3, { bdata3 } )
#pragma data(bdata3)
__export const char MAT_W3[11264] = {
#embed 11264 0 "l0_w3.bin"
};
#pragma data(data)

#pragma section( bdata4, 0 )
#pragma region( bank4, 0x8000, 0xc000, , 4, { bdata4 } )
#pragma data(bdata4)
__export const char MAT_W2[11264] = {
#embed 11264 0 "l0_w2.bin"
};
#pragma data(data)

#define DIM 128
#define HID 352
#define QB  10

int PL[176][16];
int x[DIM], xb[DIM], q[DIM], kk[16], vv[16], ATT[DIM], o[DIM];
int h1[HID], h3[HID];
int xq[HID];              /* int, non signed char: vedi nota 3 */

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

static int errori = 0;

static void controlla(const char * nome, const int * v, int n, long atteso)
{
    long s = 0;
    int i;
    for (i = 0; i < n; i++) s += (long)v[i];
    put(nome); put(p" ");
    putnum(s);
    if (s != atteso) { put(p" != "); putnum(atteso); errori++; }
    nl();
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
    int p;
    for (p = 0; p < ncoppie; p++) {
        int xa = xq[2 * p];
        int xbb = xq[2 * p + 1];
        int * t = PL[p];
        t[0] = -xa - xbb;  t[1] = -xbb;  t[2]  = xa - xbb;
        t[4] = -xa;        t[5] = 0;     t[6]  = xa;
        t[8] = -xa + xbb;  t[9] = xbb;   t[10] = xa + xbb;
    }
}

static void matvec(const char * w, int righe, int bpr, int * y,
                   long mant, char shmat, char sx)
{
    int i, k, p;
    long t;
    char sh = shmat - sx;
    for (i = 0; i < righe; i++) {
        const char * r = w + (long)i * bpr;
        int acc = 0;
        p = 0;
        for (k = 0; k < bpr; k++) {
            unsigned char b = r[k];
            acc += PL[p][b & 15];
            acc += PL[p + 1][(b >> 4) & 15];
            p += 2;
        }
        t = (long)acc;
        t = t * mant;
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

int main(void)
{
    int i, h, d;
    char s;
    long t;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"layer 0 completo"); nl(); nl();

    for (i = 0; i < DIM; i++) {
        t = (long)EMB1[i];
        t = t * EMB_MANT;
        t = t >> (EMB_SHIFT - QB);
        if (t >  32767L) t =  32767L;
        if (t < -32768L) t = -32768L;
        x[i] = (int)t;
    }
    controlla(p"x0 ", x, DIM, 1155L);

    rmsnorm(xb, x, NRM_AN, DIM);
    controlla(p"xb ", xb, DIM, 11251L);

    s = quant_shift(xb, DIM);
    build(64);
    eflash.bank = 1;
    matvec(MAT_WQ, DIM, 32, q,  WQ_MANT, WQ_SHIFT, s);
    matvec(MAT_WK, 16,  32, kk, WK_MANT, WK_SHIFT, s);
    matvec(MAT_WV, 16,  32, vv, WV_MANT, WV_SHIFT, s);
    controlla(p"q  ", q,  DIM, -1116L);
    controlla(p"k  ", kk, 16,  757L);
    controlla(p"v  ", vv, 16,  241L);

    for (h = 0; h < 8; h++)
        for (d = 0; d < 16; d++) ATT[h * 16 + d] = vv[d];

    s = quant_shift(ATT, DIM);
    build(64);
    matvec(MAT_WO, DIM, 32, o, WO_MANT, WO_SHIFT, s);
    controlla(p"o  ", o, DIM, -774L);

    for (i = 0; i < DIM; i++) {
        t = (long)x[i] + (long)o[i];
        if (t >  32767L) t =  32767L;
        if (t < -32768L) t = -32768L;
        x[i] = (int)t;
    }

    rmsnorm(xb, x, NRM_FN, DIM);
    controlla(p"xb2", xb, DIM, 4250L);

    s = quant_shift(xb, DIM);
    build(64);
    eflash.bank = 2;
    matvec(MAT_W1, HID, 32, h1, W1_MANT, W1_SHIFT, s);
    eflash.bank = 3;
    matvec(MAT_W3, HID, 32, h3, W3_MANT, W3_SHIFT, s);

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
    controlla(p"sw ", h1, HID, 10395L);

    s = quant_shift(h1, HID);
    build(176);
    eflash.bank = 4;
    matvec(MAT_W2, DIM, 88, o, W2_MANT, W2_SHIFT, s);
    controlla(p"o2 ", o, DIM, 7660L);

    for (i = 0; i < DIM; i++) {
        t = (long)x[i] + (long)o[i];
        if (t >  32767L) t =  32767L;
        if (t < -32768L) t = -32768L;
        x[i] = (int)t;
    }
    controlla(p"x2 ", x, DIM, 8041L);

    nl();
    if (errori == 0) put(p"LAYER 0 OK");
    else { put(p"ERRORI: "); putnum((long)errori); }

    while (true)
        ;
    return 0;
}
'''

open("l0full.c", "w", encoding="utf-8").write(sorgente)
print("l0full.c scritto")