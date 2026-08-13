"""gen_l0att.py - genera l0attdata.h e l0att.c: layer 0 CON ATTENZIONE sul C64.

Verifica RoPE, cache KV, punteggi, softmax e media pesata su NPOS posizioni
consecutive, dove a posizione 0 erano tutti identita' o casi banali.
A posizione 1 la softmax pesa due termini; il caso che conta e' a contesto
lungo, dove l'errore della cache int8 si accumula su decine di posizioni.

I bersagli non sono scritti a mano: si leggono da  oracolo_att.exe -dump,
cosi' oracolo e programma non possono divergere.

⚠️ Nomi delle globali: mai W1..W4, WW, L1, E0, W16, W32, LB1, LS0, LS1, WS1,
   zero - collidono con le etichette dell'assembly del runtime di oscar64 e la
   moltiplicazione a 32 bit restituisce spazzatura in silenzio (§11-bis).
"""

import re
import struct
import subprocess
import sys

D = 128
KVDIM = 16
HALF = 8
SEQ = 64
NPOS = 16          # posizioni da verificare; 16 righe stanno nello schermo

STADI = ["x0", "xb", "q", "k", "v", "qr", "kr", "kc", "vc",
         "sc", "sm", "cat", "o", "xb2", "sw", "o2", "x2"]

# ------------------------------------------------ bersagli letti dall'oracolo
try:
    uscita = subprocess.run(["./oracolo_att.exe", "-dump"],
                            capture_output=True, text=True, check=True).stdout
except (OSError, subprocess.CalledProcessError) as e:
    print("non riesco a eseguire oracolo_att.exe -dump:", e)
    print("compilalo con:  gcc -O2 -o oracolo_att.exe oracolo_att.c -lm")
    sys.exit(1)

blocchi = re.split(r"--- layer 0, posizione (\d+), token (\d+) ---", uscita)
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

print("bersagli letti per %d posizioni, token %s"
      % (NPOS, [token[p] for p in range(NPOS)]))
print("  x2 per posizione: %s" % [bersagli[p]["x2"] for p in range(NPOS)])

# ---------------------------------------------------------------- embedding
mod = open("model_ternary.bin", "rb").read()
emb = mod[len(mod) - 512 * D:]


def riga_emb(tok):
    r = emb[tok * D:(tok + 1) * D]
    return [b - 256 if b > 127 else b for b in r]


# ------------------------------------------------------------------ tabelle
t = open("tables.bin", "rb").read()
texp = struct.unpack_from("<512h", t, 256 * 2)
base_cos = (256 + 512 + 256) * 2
base_sin = base_cos + SEQ * HALF * 2
cos = [struct.unpack_from("<%dh" % HALF, t, base_cos + p * HALF * 2)
       for p in range(NPOS)]
sin = [struct.unpack_from("<%dh" % HALF, t, base_sin + p * HALF * 2)
       for p in range(NPOS)]

# ------------------------------------------------------------ l0attdata.h
with open("l0attdata.h", "w", encoding="utf-8") as g:
    g.write("/* generato da gen_l0att.py */\n")
    g.write("#define NPOS %d\n" % NPOS)
    g.write("#define NSTADI %d\n\n" % len(STADI))
    g.write("/* stadi, in ordine: %s */\n" % " ".join(STADI))
    g.write("const int EMBT[%d][128] = {\n" % NPOS)
    for p in range(NPOS):
        g.write("{" + ",".join(map(str, riga_emb(token[p]))) + "},\n")
    g.write("};\n")
    g.write("const int TEXP[512] = {" + ",".join(map(str, texp)) + "};\n")
    g.write("const int RCOS[%d][8] = {\n" % NPOS)
    for p in range(NPOS):
        g.write("{" + ",".join(map(str, cos[p])) + "},\n")
    g.write("};\n")
    g.write("const int RSIN[%d][8] = {\n" % NPOS)
    for p in range(NPOS):
        g.write("{" + ",".join(map(str, sin[p])) + "},\n")
    g.write("};\n")
    g.write("\n/* bersagli, da oracolo_att.exe -dump */\n")
    g.write("const long BERS[%d][%d] = {\n" % (NPOS, len(STADI)))
    for p in range(NPOS):
        g.write("{" + ",".join("%dL" % bersagli[p][s] for s in STADI) + "},\n")
    g.write("};\n")
print("l0attdata.h scritto")

# ------------------------------------------------------------------ l0att.c
sorgente = r'''/* l0att.c - GENERATO DA gen_l0att.py.  Layer 0 CON ATTENZIONE sul C64.

   Verifica su NPOS posizioni consecutive quello che a posizione 0 era banale:
   RoPE, cache KV, punteggi, softmax, media pesata.  Ogni stadio di ogni
   posizione e' confrontato con l'oracolo: NPOS x NSTADI confronti.

   Accorgimenti oscar64 (vedi documento di progetto):
     1. ogni shift a 32 bit spezzato in un passaggio separato su una long
     2. clamp scritto inline, nessun parametro long passato a funzioni
     3. niente signed char nell'aritmetica
     4. NESSUNA globale chiamata W1..W4, WW, L1, E0, W16, W32, LB1, LS0,
        LS1, WS1, zero: sono etichette del runtime.  Matrici = MAT_*  (§11-bis)

   ⚠️ La cache qui e' in int per non toccare l'estensione di segno: nel
   programma vero va a int8, che e' il presupposto del bilancio RAM (§14).
   I VALORI sono comunque quelli int8, arrotondati in scrittura.

   Compilare ed eseguire:  .\c64.ps1 l0att -Cicli 800000000
*/

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include "l0fdata.h"
#include "l0attdata.h"

/* niente heap: il programma non alloca, e con NPOS posizioni di bersagli lo
   spazio serve tutto.  Con heap incluso oscar64 da' "error 3034: Cannot
   place heap section". */
#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, stack } )

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

#define DIM   128
#define HID   352
#define QB    10
#define HEADS 8
#define HDIM  16
#define KVD   16
#define KSH   7
#define VSH   5

int PL[176][16];
int x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], cat[DIM], o[DIM];
int h1[HID], h3[HID];
int xq[HID];
int KC[NPOS][KVD], VC[NPOS][KVD];
long att[NPOS], peso[NPOS];
long sc_somma, sm_somma;

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

static int  errori = 0, confronti = 0;
static int  pr_pos = -1, pr_st = -1;
static long pr_ott = 0, pr_att = 0;

/* confronta in silenzio e ricorda il PRIMO scostamento */
static void verifica(int pos, int st, long s)
{
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
    int p;
    for (p = 0; p < ncoppie; p++) {
        int xa  = xq[2 * p];
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

/* RoPE: ruota le coppie ADIACENTI (0,1) (2,3) ... con le frequenze 0..7 */
static void rope(int * p, int pos)
{
    int i;
    long a, bq, r, im;
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

/* attenzione su t = 0..pos.  MQA: una sola testa KV, condivisa dalle otto.
   Normalizzazione con UN reciproco per testa, non una divisione per
   elemento: 32 divisioni per token invece di 512, a parita' di risultato. */
static void attenzione(int pos)
{
    int h, d, t;
    long s, best, somma, inv;
    sc_somma = 0;
    sm_somma = 0;
    for (h = 0; h < HEADS; h++) {
        best = -2147483647L;
        for (t = 0; t <= pos; t++) {
            s = 0;
            for (d = 0; d < HDIM; d++)
                s += (long)q[h * HDIM + d] * (long)KC[t][d];
            s = s >> (QB + 2 - KSH);
            att[t] = s;
            if (s > best) best = s;
            sc_somma += s;
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
        for (t = 0; t <= pos; t++) {
            s = att[t] * inv;
            s = s >> 15;
            peso[t] = s;
            sm_somma += s;
        }
        for (d = 0; d < HDIM; d++) {
            s = 0;
            for (t = 0; t <= pos; t++)
                s += peso[t] * (long)VC[t][d];
            s = s >> (15 - VSH);
            if (s >  32767L) s =  32767L;
            if (s < -32768L) s = -32768L;
            cat[h * HDIM + d] = (int)s;
        }
    }
}

int main(void)
{
    int i, pos, d;
    char s;
    long t;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"attenzione, "); putnum((long)NPOS); put(p" posizioni"); nl();

    for (pos = 0; pos < NPOS; pos++) {

        for (i = 0; i < DIM; i++) {
            t = (long)EMBT[pos][i];
            t = t * EMB_MANT;
            t = t >> (EMB_SHIFT - QB);
            if (t >  32767L) t =  32767L;
            if (t < -32768L) t = -32768L;
            x[i] = (int)t;
        }
        verifica_vec(pos, 0, x, DIM);

        rmsnorm(xb, x, NRM_AN, DIM);
        verifica_vec(pos, 1, xb, DIM);

        s = quant_shift(xb, DIM);
        build(64);
        eflash.bank = 1;
        matvec(MAT_WQ, DIM, 32, q,  WQ_MANT, WQ_SHIFT, s);
        matvec(MAT_WK, KVD, 32, kk, WK_MANT, WK_SHIFT, s);
        matvec(MAT_WV, KVD, 32, vv, WV_MANT, WV_SHIFT, s);
        verifica_vec(pos, 2, q,  DIM);
        verifica_vec(pos, 3, kk, KVD);
        verifica_vec(pos, 4, vv, KVD);

        for (i = 0; i < HEADS; i++) rope(q + i * HDIM, pos);
        rope(kk, pos);
        verifica_vec(pos, 5, q,  DIM);
        verifica_vec(pos, 6, kk, KVD);

        /* scrittura in cache: valori int8, ARROTONDANDO.
           Col troncamento la sequenza dei 35 token diverge (§14). */
        for (d = 0; d < KVD; d++) {
            t = (long)kk[d] + 64L;         /* 1 << (KSH-1) */
            t = t >> KSH;
            if (t >  127L) t =  127L;
            if (t < -128L) t = -128L;
            KC[pos][d] = (int)t;
            t = (long)vv[d] + 16L;         /* 1 << (VSH-1) */
            t = t >> VSH;
            if (t >  127L) t =  127L;
            if (t < -128L) t = -128L;
            VC[pos][d] = (int)t;
        }
        verifica_vec(pos, 7, KC[pos], KVD);
        verifica_vec(pos, 8, VC[pos], KVD);

        attenzione(pos);
        verifica(pos, 9,  sc_somma);
        verifica(pos, 10, sm_somma);
        verifica_vec(pos, 11, cat, DIM);

        s = quant_shift(cat, DIM);
        build(64);
        eflash.bank = 1;
        matvec(MAT_WO, DIM, 32, o, WO_MANT, WO_SHIFT, s);
        verifica_vec(pos, 12, o, DIM);

        for (i = 0; i < DIM; i++) {
            t = (long)x[i] + (long)o[i];
            if (t >  32767L) t =  32767L;
            if (t < -32768L) t = -32768L;
            x[i] = (int)t;
        }

        rmsnorm(xb, x, NRM_FN, DIM);
        verifica_vec(pos, 13, xb, DIM);

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
        verifica_vec(pos, 14, h1, HID);

        s = quant_shift(h1, HID);
        build(176);
        eflash.bank = 4;
        matvec(MAT_W2, DIM, 88, o, W2_MANT, W2_SHIFT, s);
        verifica_vec(pos, 15, o, DIM);

        for (i = 0; i < DIM; i++) {
            t = (long)x[i] + (long)o[i];
            if (t >  32767L) t =  32767L;
            if (t < -32768L) t = -32768L;
            x[i] = (int)t;
        }
        verifica_vec(pos, 16, x, DIM);

        /* una riga per posizione: x2 dipende da tutto quel che sta a monte */
        put(p"p"); putnum((long)pos); put(p" ");
        t = 0;
        for (i = 0; i < DIM; i++) t += (long)x[i];
        putnum(t);
        if (t != BERS[pos][16]) { put(p" != "); putnum(BERS[pos][16]); }
        nl();
    }

    putnum((long)confronti); put(p" confronti, ");
    putnum((long)errori); put(p" errori"); nl();
    if (errori == 0) {
        put(p"ATTENZIONE OK");
    } else {
        put(p"primo: p"); putnum((long)pr_pos);
        put(p" st");      putnum((long)pr_st);
        put(p" ");        putnum(pr_ott);
        put(p" != ");     putnum(pr_att);
    }

    while (true)
        ;
    return 0;
}
'''

open("l0att.c", "w", encoding="utf-8").write(sorgente)
print("l0att.c scritto (%d posizioni x %d stadi = %d confronti)"
      % (NPOS, len(STADI), NPOS * len(STADI)))
