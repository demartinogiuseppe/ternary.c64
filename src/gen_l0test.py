"""gen_l0test.py — genera l0test.c, il primo pezzo di assemblaggio vero.

Cosa verifica:
  - embedding -> x, con la scala in (mantissa, shift)
  - RMSnorm in virgola fissa con la lookup di 1/sqrt
  - la tabella PL/PH costruita UNA VOLTA e usata per tre matrici diverse
    (wq, wk, wv ricevono lo stesso xb: e' l'ottimizzazione trovata prima)
  - tre matrici di forme diverse (128x128, 16x128) nello stesso banco
  - i riscalamenti con MANT/SHIFT per matrice

Bersagli da oracolo_l0.exe:
    xb = 11251   q = -1116   k = 757   v = 241

Kernel in C, non in assembly: qui si verifica la correttezza
dell'assemblaggio, non la velocita'. L'assembly e' gia' validato.
"""

RIGHE_Q, RIGHE_KV, COL = 128, 16, 128
BPR = COL // 4          # 32 byte per riga

sorgente = f'''/* l0test.c — GENERATO DA gen_l0test.py.
   Primo assemblaggio: embedding, RMSnorm, tre matvec da un banco.

   Compilare:  oscar64 -O2 -tf=crt -o=l0test.crt l0test.c
   (Detach in VICE PRIMA di ricompilare)

   Bersagli: xb=11251  q=-1116  k=757  v=241 */

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include "l0data.h"

#pragma region( main, 0x0900, 0x8000, , , {{ code, data, bss, heap, stack }} )

/* banco 1: le tre matrici del layer 0 che condividono lo stesso input */
#pragma section( bdata1, 0 )
#pragma region( bank1, 0x8000, 0xc000, , 1, {{ bdata1 }} )
#pragma data(bdata1)
__export const char WQ[{RIGHE_Q * BPR}] = {{
#embed {RIGHE_Q * BPR} 0 "l0_wq.bin"
}};
__export const char WK[{RIGHE_KV * BPR}] = {{
#embed {RIGHE_KV * BPR} 0 "l0_wk.bin"
}};
__export const char WV[{RIGHE_KV * BPR}] = {{
#embed {RIGHE_KV * BPR} 0 "l0_wv.bin"
}};
#pragma data(data)

#define DIM {COL}
#define QB  10

/* tabella 1/sqrt: terza di tables.bin, dopo silu (256) ed exp (512) */
int T_rsq[256];

int PL[64][16];          /* interi: qui conta la correttezza, non lo spazio */
int x[DIM], xb[DIM], q[DIM], kk[16], vv[16];
signed char xq[DIM];

static CharWin cw;

static void put(const char * s) {{ cwin_put_string(&cw, s, 7); }}
static void nl(void)            {{ cwin_cursor_newline(&cw); }}

static void putnum(long v)
{{
    char buf[12];
    char i = 12;
    unsigned long u;
    if (v < 0) {{ cwin_put_char(&cw, p'-', 7); u = (unsigned long)(-v); }}
    else       u = (unsigned long)v;
    if (u == 0) {{ cwin_put_char(&cw, p'0', 7); return; }}
    while (u) {{ buf[--i] = p'0' + (char)(u % 10); u /= 10; }}
    while (i < 12) cwin_put_char(&cw, buf[i++], 7);
}}

static int clampfx(long v)
{{
    if (v >  32767L) return  32767;
    if (v < -32768L) return -32768;
    return (int)v;
}}

/* quantizza a int8 con uno shift, troncamento */
static char quant_shift(const int * v, int n)
{{
    long mx = 0;
    int i;
    char s = 0;
    for (i = 0; i < n; i++) {{
        long a = v[i] < 0 ? -(long)v[i] : (long)v[i];
        if (a > mx) mx = a;
    }}
    while ((mx >> s) > 127) s++;
    for (i = 0; i < n; i++) xq[i] = (signed char)(v[i] >> s);
    return s;
}}

/* costruisce PL dai valori gia' quantizzati in xq */
static void build(void)
{{
    int p;
    for (p = 0; p < 64; p++) {{
        int xa = xq[2 * p];
        int xb2 = xq[2 * p + 1];
        int * t = PL[p];
        t[0]  = -xa - xb2;  t[1] = -xb2;  t[2]  = xa - xb2;
        t[4]  = -xa;        t[5] = 0;     t[6]  = xa;
        t[8]  = -xa + xb2;  t[9] = xb2;   t[10] = xa + xb2;
    }}
}}

/* y = W x, con la tabella gia' costruita e la scala della matrice */
static void matvec(const char * w, int righe, int * y,
                   unsigned mant, char shmat, char sx)
{{
    int i, k, p;
    char sh = shmat - sx;
    for (i = 0; i < righe; i++) {{
        const char * r = w + i * {BPR};
        int acc = 0;
        p = 0;
        for (k = 0; k < {BPR}; k++) {{
            unsigned char b = r[k];
            acc += PL[p][b & 15];
            acc += PL[p + 1][(b >> 4) & 15];
            p += 2;
        }}
        y[i] = clampfx(((long)acc * mant) >> sh);
    }}
}}

static void rmsnorm(int * o, const int * v, const int * w, int n)
{{
    long ss = 0;
    int i, e = 0, k;
    long inv;
    for (i = 0; i < n; i++) ss += (long)v[i] * v[i];
    ss /= n;
    if (ss < 1) ss = 1;
    while (ss <  (1L << 20)) {{ ss <<= 2; e++; }}
    while (ss >= (1L << 22)) {{ ss >>= 2; e--; }}
    k = (int)(((ss - (1L << 20)) * 256L) / (3L << 20));
    if (k > 255) k = 255;
    inv = (long)T_rsq[k] << e;
    for (i = 0; i < n; i++) {{
        long t = ((long)v[i] * inv) >> 14;
        o[i] = clampfx((t * w[i]) >> 12);
    }}
}}

int main(void)
{{
    int i;
    long sxb, sq, sk, sv;
    char s;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"assemblaggio layer 0"); nl(); nl();

    /* la tabella 1/sqrt e' compilata dentro l0data.h come RSQ[] */
    for (i = 0; i < 256; i++) T_rsq[i] = RSQ[i];

    /* embedding del token BOS = 1 */
    for (i = 0; i < DIM; i++)
        x[i] = clampfx(((long)EMB1[i] * EMB_MANT) >> (EMB_SHIFT - QB));

    rmsnorm(xb, x, NRM_AN, DIM);

    /* UNA sola quantizzazione e UNA sola tabella per tre matrici */
    s = quant_shift(xb, DIM);
    build();

    eflash.bank = 1;
    matvec(WQ, {RIGHE_Q},  q,  WQ_MANT, WQ_SHIFT, s);
    matvec(WK, {RIGHE_KV}, kk, WK_MANT, WK_SHIFT, s);
    matvec(WV, {RIGHE_KV}, vv, WV_MANT, WV_SHIFT, s);

    sxb = sq = sk = sv = 0;
    for (i = 0; i < DIM; i++) {{ sxb += xb[i]; sq += q[i]; }}
    for (i = 0; i < 16; i++)  {{ sk += kk[i]; sv += vv[i]; }}

    put(p"xb "); putnum(sxb); put(p"  atteso 11251"); nl();
    put(p"q  "); putnum(sq);  put(p"  atteso -1116"); nl();
    put(p"k  "); putnum(sk);  put(p"  atteso 757"); nl();
    put(p"v  "); putnum(sv);  put(p"  atteso 241"); nl(); nl();

    put(p"shift attivazioni: "); putnum((long)s); nl(); nl();

    if (sxb == 11251L && sq == -1116L && sk == 757L && sv == 241L)
        put(p"ASSEMBLAGGIO OK");
    else
        put(p"DIVERSO");

    while (true)
        ;
    return 0;
}}
'''

open("l0test.c", "w").write(sorgente)
print("l0test.c scritto")

# aggiunge la tabella 1/sqrt a l0data.h, presa da tables.bin
import struct
t = open("tables.bin", "rb").read()
rsq = struct.unpack_from("<256h", t, (256 + 512) * 2)
with open("l0data.h", "a") as g:
    g.write("\nconst int RSQ[256] = {" + ",".join(map(str, rsq)) + "};\n")
print("RSQ[256] aggiunta a l0data.h")
print("bersagli: xb=11251  q=-1116  k=757  v=241")