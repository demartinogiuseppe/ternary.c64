"""gen_l0kern.py - genera l0kern.c: come l0att.c ma con il KERNEL MAC IN
ASSEMBLY al posto del matvec in C, e il cronometro CIA su build e calcolo.

Perche' questo passo.  Il matvec in C e' il collo di bottiglia: 16 posizioni
hanno richiesto ~800 M cicli in VICE.  Il kernel a 21 cicli/MAC e' pronto e
verificato (cart4.c, verifica X), ma non e' mai stato innestato nella
pipeline vera.  Qui si innesta e si cronometra, tenendo tutti e 17 gli stadi
di verifica su NPOS posizioni: se un numero cambia, il kernel e' sbagliato.

TRE COSE CAMBIANO rispetto a l0att.c, nessun'altra:

  1. PL[176][16] di int  ->  PL[2816] e PH[2816] di unsigned char.
     L'assembly somma a 8 bit con riporto, quindi byte bassi e alti stanno in
     due tabelle separate.  L'indirizzo della coppia p, nibble n e'
     (p >> 4)*256 + (p & 15)*16 + n, che vale esattamente p*16 + n: la
     tabella e' lineare, la spezzatura in blocchi da 256 serve solo perche'
     Y e' a 8 bit e il blocco diventa un offset costante sull'assoluto.
     Occupazione identica: 2 x 2816 = 5632 B, come 176 x 16 int.

  2. matvec in C -> quattro kernel srotolati, UNO PER BANCO.
     I kernel NON stanno in RAM: l0att ha ~1,9 KB liberi e i due srotolati
     pesano 1620 B (32 byte/riga) e 4406 B (88 byte/riga).  Vanno nel banco
     INSIEME ai pesi che leggono, perche' codice e dati condividono la
     finestra $8000 (sec. 12).  Meccanismo gia' verificato in cart8.c.

  3. cache KV in int -> signed char, cioe' la cache a int8 VERA.
     In l0att era in int per non toccare l'estensione di segno di oscar64: i
     valori erano gia' quelli int8, ma occupavano il doppio.  Il bilancio RAM
     (sec. 14) sta in piedi solo se la cache occupa davvero 8192 B invece di
     16384 su quattro layer a SEQ 64, quindi va provato prima di appoggiarci
     sopra i layer.  ⚠️ signed char esplicito: in oscar64 char e' SENZA segno.

Pagina zero usata dall'assembly: $02 temporaneo, $F2/$F3 accumulatore,
$F4/$F5 puntatore alla riga.  Il compilatore con -O2 tiene le sue globali
sopra $F7 (regione zeropage 00f7-00ff nel .map) e i suoi registri interni
sotto $30: nessuna sovrapposizione.  ATTENZIONE: con -O3 si riattiva -Oz e
la collisione torna (sec. 11).

I bersagli sono quelli di l0attdata.h, generato da gen_l0att.py leggendo
oracolo_att.exe -dump.  Rigenerare quello PRIMA di questo se cambia
l'oracolo.

Uso:  python gen_l0kern.py
      .\\c64.ps1 l0kern -Cicli 200000000
"""

import os
import re
import sys

# ------------------------------------------------- controllo dei prerequisiti
if not os.path.exists("l0attdata.h"):
    print("manca l0attdata.h: esegui prima  python gen_l0att.py")
    sys.exit(1)

testa = open("l0attdata.h", encoding="utf-8").read(400)
m = re.search(r"#define NPOS (\d+)", testa)
if not m:
    print("l0attdata.h non dichiara NPOS")
    sys.exit(1)
NPOS = int(m.group(1))
print("bersagli da l0attdata.h, NPOS = %d" % NPOS)

# ---------------------------------------------------------- kernel in assembly
# Identico a quello di gen_cart_asm.py (verifica X, 21 cicli/MAC), con le
# etichette rese uniche perche' qui ce ne sono quattro copie nello stesso
# programma.
PTR_LO, PTR_HI = "$f4", "$f5"
ACC_LO, ACC_HI = "$f2", "$f3"
TMP = "$02"


def blocco(j):
    """le due meta' del byte j della riga: due letture in tabella, niente mul"""
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
    """corpo assembly di una riga da bpr byte; il puntatore avanza da solo"""
    asm = ["        lda #0", "        sta %s" % ACC_LO, "        sta %s" % ACC_HI]
    for j in range(bpr):
        asm.append("        // byte %d" % j)
        asm += blocco(j)
    asm += ["        lda %s" % ACC_LO, "        sta rowres",
            "        lda %s" % ACC_HI, "        sta rowres + 1",
            "        // avanza di %d byte" % bpr,
            "        clc", "        lda %s" % PTR_LO, "        adc #%d" % bpr,
            "        sta %s" % PTR_LO, "        bcc %s" % etichetta,
            "        inc %s" % PTR_HI,
            "%s:" % etichetta]
    return "\n".join(asm)


# banco -> (byte per riga, matrici che ci stanno dentro)
BANCHI = [(1, 32, "MAT_WQ, MAT_WK, MAT_WV, MAT_WO"),
          (2, 32, "MAT_W1"),
          (3, 32, "MAT_W3"),
          (4, 88, "MAT_W2")]

MATRICI = {
    1: [("MAT_WQ", 4096, "l0_wq.bin"), ("MAT_WK", 512, "l0_wk.bin"),
        ("MAT_WV", 512, "l0_wv.bin"), ("MAT_WO", 4096, "l0_wo.bin")],
    2: [("MAT_W1", 11264, "l0_w1.bin")],
    3: [("MAT_W3", 11264, "l0_w3.bin")],
    4: [("MAT_W2", 11264, "l0_w2.bin")],
}

pezzi = []
for banco, bpr, quali in BANCHI:
    pezzi.append("/* ---------------- banco %d: kernel %d byte/riga + %s */"
                 % (banco, bpr, quali))
    pezzi.append("#pragma code ( bcode%d )" % banco)
    pezzi.append("#pragma data ( bdata%d )" % banco)
    pezzi.append("")
    for nome, n, f in MATRICI[banco]:
        pezzi.append("__export const char %s[%d] = {" % (nome, n))
        pezzi.append("#embed %d 0 \"%s\"" % (n, f))
        pezzi.append("};")
    pezzi.append("")
    pezzi.append("__noinline void macrow_b%d(void)" % banco)
    pezzi.append("{")
    pezzi.append("    __asm")
    pezzi.append("    {")
    pezzi.append(kernel(bpr, "av_b%d" % banco))
    pezzi.append("    }")
    pezzi.append("}")
    pezzi.append("")

pezzi.append("#pragma code ( code )")
pezzi.append("#pragma data ( data )")
BANCHI_C = "\n".join(pezzi)

REGIONI = "\n".join(
    "#pragma section( bcode%d, 0 )\n"
    "#pragma section( bdata%d, 0 )\n"
    "#pragma region( bank%d, 0x8000, 0xc000, , %d, { bcode%d, bdata%d } )"
    % (b, b, b, b, b, b) for b, _, _ in BANCHI)

# ----------------------------------------------------------------- l0kern.c
sorgente = r'''/* l0kern.c - GENERATO DA gen_l0kern.py.  Layer 0 con attenzione, KERNEL MAC
   IN ASSEMBLY dai banchi di cartuccia, cronometrato col timer B della CIA 2.

   Stessi 17 stadi di verifica di l0att.c su NPOS posizioni: il kernel non
   deve cambiare NEMMENO UN NUMERO.  Se cambia, e' sbagliato.

   Rispetto a l0att.c cambiano solo tre cose:
     - PL/PH sono due tabelle di byte (l'adc dell'assembly e' a 8 bit)
     - matvec chiama un kernel srotolato per banco invece di sommare in C
     - la cache KV e' a signed char, cioe' int8 vera: 8192 B invece di 16384
       su quattro layer a SEQ 64, che e' quello su cui poggia il bilancio RAM

   ⚠️ COMPILARE CON -O2, NON -O3: -O3 include -Oz, che piazza le globali in
   pagina zero e collide con $F2-$F5 usati dall'assembly (sec. 11).

   Compilare ed eseguire:  .\c64.ps1 l0kern -Cicli 200000000
*/

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include "l0fdata.h"
#include "l0attdata.h"

/* niente heap: il programma non alloca (sec. errore 3034).
   La pila di default e' 4 KB e non ci sta: l0att lasciava 384 byte liberi.
   Il bilancio RAM (sec. 14) prevede 1024 byte di pila; qui 2048 per margine.
   Se oscar64 dice "Cannot place stack section", la prima leva e' questa. */
#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, stack } )
#pragma stacksize( 2048 )

__REGIONI__

#define DIM   128
#define HID   352
#define QB    10
#define HEADS 8
#define HDIM  16
#define KVD   16
#define KSH   7
#define VSH   5

/* Condivise fra RAM e banchi.  PL/PH: byte bassi e alti delle nove somme
   ±xa ±xb, 16 voci per coppia, 176 coppie.  L'indice (p>>4)*256+(p&15)*16+n
   e' lineare in p*16+n; la spezzatura in blocchi da 256 esiste solo perche'
   Y e' a 8 bit.  rowres: uscita di una riga, scritta dall'assembly. */
unsigned char PL[2816];
unsigned char PH[2816];
int rowres;

__BANCHI__

int x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], cat[DIM], o[DIM];
int h1[HID], h3[HID];
int xq[HID];

/* Cache KV a int8 VERA, non int: e' il presupposto del bilancio RAM (sec.
   14), 8192 B invece di 16384 su quattro layer a SEQ 64.  In l0att era in
   int per non toccare l'estensione di segno di oscar64; qui la si tocca, ed
   e' esattamente la cosa che questa verifica deve dimostrare.
   ⚠️ signed char esplicito: in oscar64 char e' senza segno. */
signed char KC[NPOS][KVD], VC[NPOS][KVD];

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

/* ------------------------------------------------------- cronometro CIA 2
   L'orologio KERNAL resta a zero in modalita' cartuccia.  Il timer B e' a
   16 bit: va interrogato almeno ogni 65536 cicli, cioe' a ogni riga. */
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

/* c_nudo: 128 righe di wq SENZA riscalamento, per confronto diretto con
   cart4 (347.682 cicli, 21/MAC).  c_wq: le stesse 128 righe dentro la
   pipeline vera, riscalamento compreso.  La differenza e' il costo
   dell'epilogo in C, che nessuna misura precedente aveva isolato. */
static unsigned long c_build = 0, c_mac = 0, c_wq = 0, c_nudo = 0;

/* -------------------------------------------------------------- verifiche */
static int  errori = 0, confronti = 0;
static int  pr_pos = -1, pr_st = -1;
static long pr_ott = 0, pr_att = 0;

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

/* La stessa cosa sulla cache a int8.  Il (long)v[i] qui NON e' un dettaglio:
   e' l'estensione di segno da signed char a 32 bit, cioe' la cosa in prova. */
static void verifica_vec8(int pos, int st, const signed char * v, int n)
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

/* Le nove somme, per simmetria: quattro addizioni e quattro negazioni, con
   i puntatori che avanzano (variante C di sec. 8, -23% sulla A). */
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

/* Il puntatore alla riga vive in $F4/$F5 e lo fa avanzare il kernel.
   banco = quale kernel chiamare: e' anche il banco in cui sta, insieme
   ai pesi che legge (codice e dati nella stessa finestra $8000). */
static void matvec(const char * w, int righe, int * y,
                   long mant, char shmat, char sx, char banco)
{
    int i;
    long t;
    char sh = shmat - sx;

    eflash.bank = banco;
    *(const char **)0x00F4 = w;

    for (i = 0; i < righe; i++) {
        switch (banco) {
        case 1:  macrow_b1(); break;
        case 2:  macrow_b2(); break;
        case 3:  macrow_b3(); break;
        default: macrow_b4(); break;
        }
        t = (long)rowres;
        t = t * mant;
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
   Un reciproco per testa, non una divisione per elemento (sec. 14). */
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
    unsigned long cc;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"kernel asm, "); putnum((long)NPOS); put(p" posizioni"); nl();

    /* kernel nudo: nessun riscalamento, come cart4.  Le tabelle sono ancora
       a zero ma il tempo non dipende dai dati: nel kernel non c'e' un solo
       salto condizionato sui valori. */
    eflash.bank = 1;
    *(const char **)0x00F4 = MAT_WQ;
    tmr_start();
    for (i = 0; i < DIM; i++) {
        macrow_b1();
        q[i] = rowres;
        tmr_poll();
    }
    c_nudo = tmr_stop();

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
        tmr_start(); build(64);           c_build += tmr_stop();
        tmr_start();
        matvec(MAT_WQ, DIM, q,  WQ_MANT, WQ_SHIFT, s, 1);
        cc = tmr_stop();
        c_wq  += cc;
        c_mac += cc;
        tmr_start();
        matvec(MAT_WK, KVD, kk, WK_MANT, WK_SHIFT, s, 1);
        matvec(MAT_WV, KVD, vv, WV_MANT, WV_SHIFT, s, 1);
        c_mac += tmr_stop();
        verifica_vec(pos, 2, q,  DIM);
        verifica_vec(pos, 3, kk, KVD);
        verifica_vec(pos, 4, vv, KVD);

        for (i = 0; i < HEADS; i++) rope(q + i * HDIM, pos);
        rope(kk, pos);
        verifica_vec(pos, 5, q,  DIM);
        verifica_vec(pos, 6, kk, KVD);

        /* cache int8, ARROTONDANDO: col troncamento la sequenza diverge */
        for (d = 0; d < KVD; d++) {
            t = (long)kk[d] + 64L;         /* 1 << (KSH-1) */
            t = t >> KSH;
            if (t >  127L) t =  127L;
            if (t < -128L) t = -128L;
            KC[pos][d] = (signed char)t;
            t = (long)vv[d] + 16L;         /* 1 << (VSH-1) */
            t = t >> VSH;
            if (t >  127L) t =  127L;
            if (t < -128L) t = -128L;
            VC[pos][d] = (signed char)t;
        }
        verifica_vec8(pos, 7, KC[pos], KVD);
        verifica_vec8(pos, 8, VC[pos], KVD);

        attenzione(pos);
        verifica(pos, 9,  sc_somma);
        verifica(pos, 10, sm_somma);
        verifica_vec(pos, 11, cat, DIM);

        s = quant_shift(cat, DIM);
        tmr_start(); build(64);           c_build += tmr_stop();
        tmr_start();
        matvec(MAT_WO, DIM, o, WO_MANT, WO_SHIFT, s, 1);
        c_mac += tmr_stop();
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
        tmr_start(); build(64);           c_build += tmr_stop();
        tmr_start();
        matvec(MAT_W1, HID, h1, W1_MANT, W1_SHIFT, s, 2);
        matvec(MAT_W3, HID, h3, W3_MANT, W3_SHIFT, s, 3);
        c_mac += tmr_stop();

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
        tmr_start(); build(176);          c_build += tmr_stop();
        tmr_start();
        matvec(MAT_W2, DIM, o, W2_MANT, W2_SHIFT, s, 4);
        c_mac += tmr_stop();
        verifica_vec(pos, 15, o, DIM);

        for (i = 0; i < DIM; i++) {
            t = (long)x[i] + (long)o[i];
            if (t >  32767L) t =  32767L;
            if (t < -32768L) t = -32768L;
            x[i] = (int)t;
        }
        verifica_vec(pos, 16, x, DIM);
    }

    putnum((long)confronti); put(p" confronti, ");
    putnum((long)errori); put(p" errori"); nl();
    if (errori == 0) {
        put(p"KERNEL ASM + CACHE INT8 OK"); nl();
    } else {
        put(p"primo: p"); putnum((long)pr_pos);
        put(p" st");      putnum((long)pr_st);
        put(p" ");        putnum(pr_ott);
        put(p" != ");     putnum(pr_att); nl();
    }
    nl();

    /* 172032 MAC ternari per posizione: wq 16384 wk 2048 wv 2048 wo 16384
       w1 45056 w3 45056 w2 45056.  Per token vero sono quattro layer. */
    put(p"build tot : "); putnum((long)(c_build / 1000UL)); put(p"k"); nl();
    put(p"mac tot   : "); putnum((long)(c_mac / 1000UL)); put(p"k"); nl();
    put(p"per pos   : "); putnum((long)(c_mac / (unsigned long)NPOS)); nl();
    put(p"cicli/mac : ");
    putnum((long)(c_mac / (172032UL * (unsigned long)NPOS))); nl();
    put(p"4 layer   : ");
    putnum((long)((c_mac / (unsigned long)NPOS) * 4UL / 1000UL));
    put(p"k per token"); nl(); nl();

    /* dove se ne vanno: kernel nudo contro riga completa */
    put(p"wq nudo   : "); putnum((long)c_nudo);
    put(p" (cart4 347682)"); nl();
    put(p"wq vero   : ");
    putnum((long)(c_wq / (unsigned long)NPOS)); nl();
    put(p"riga nuda : "); putnum((long)(c_nudo / (unsigned long)DIM)); nl();
    put(p"riga vera : ");
    putnum((long)(c_wq / ((unsigned long)NPOS * (unsigned long)DIM))); nl();
    put(p"epilogo   : ");
    putnum((long)(c_wq / ((unsigned long)NPOS * (unsigned long)DIM)
                  - c_nudo / (unsigned long)DIM));
    put(p" per riga"); nl();

    while (true)
        ;
    return 0;
}
'''

sorgente = sorgente.replace("__REGIONI__", REGIONI).replace("__BANCHI__", BANCHI_C)
open("l0kern.c", "w", encoding="utf-8").write(sorgente)

righe_asm = sum(kernel(bpr, "av").count("\n") + 1 for _, bpr, _ in BANCHI)
print("l0kern.c scritto: 4 kernel, %d righe di assembly in tutto" % righe_asm)
print("COMPILARE CON -O2, NON -O3")
