/* cart6.c — ottimizzazione della costruzione delle tabelle PL/PH.

   La build costa ~26.000 cicli e va rifatta per ogni prodotto matrice-vettore:
   112 volte per token = 2,9 M cicli, il 13% del budget. Emerso solo quando il
   cronometro e' passato dai jiffy ai cicli singoli.

   Tre varianti misurate nello stesso programma:
     A  come adesso: 9 voci calcolate, indici a 16 bit su array da 1024
     B  sfrutta la simmetria: 4 somme + 4 negazioni + 1 zero
     C  come B, ma con puntatori che avanzano invece di ricalcolare base

   Bersaglio: i risultati devono restare identici fra le tre, e il kernel
   deve continuare a dare somma -17721.

   Compilare:  oscar64 -O2 -tf=crt -o=cart6.crt cart6.c
   (Detach la cartuccia in VICE PRIMA di ricompilare!) */

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include "c64data.h"

#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, heap, stack } )

unsigned char PL[1024];
unsigned char PH[1024];
unsigned char QL[1024];      /* copia di controllo */
unsigned char QH[1024];

static CharWin cw;
static volatile byte scarto;
static unsigned giri;

static void put(const char * s) { cwin_put_string(&cw, s, 7); }
static void nl(void)            { cwin_cursor_newline(&cw); }

static void putlong(unsigned long v)
{
    char buf[12];
    char i = 12;
    if (v == 0) { cwin_put_char(&cw, p'0', 7); return; }
    while (v) { buf[--i] = p'0' + (char)(v % 10); v /= 10; }
    while (i < 12) cwin_put_char(&cw, buf[i++], 7);
}

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

/* ---------- A: la versione attuale ---------- */
static void buildA(void)
{
    int p, v, base;
    for (p = 0; p < 64; p++) {
        int xa = XIN[2 * p];
        int xb = XIN[2 * p + 1];
        base = (p >> 4) * 256 + (p & 15) * 16;
        v = -xa - xb; PL[base]      = v; PH[base]      = v >> 8;
        v = -xb;      PL[base + 1]  = v; PH[base + 1]  = v >> 8;
        v =  xa - xb; PL[base + 2]  = v; PH[base + 2]  = v >> 8;
        v = -xa;      PL[base + 4]  = v; PH[base + 4]  = v >> 8;
                      PL[base + 5]  = 0; PH[base + 5]  = 0;
        v =  xa;      PL[base + 6]  = v; PH[base + 6]  = v >> 8;
        v = -xa + xb; PL[base + 8]  = v; PH[base + 8]  = v >> 8;
        v =  xb;      PL[base + 9]  = v; PH[base + 9]  = v >> 8;
        v =  xa + xb; PL[base + 10] = v; PH[base + 10] = v >> 8;
    }
}

/* ---------- B: sfrutta la simmetria ----------
   Le 9 voci sono +-xa +-xb. Quattro sono il negato di altre quattro,
   la nona e' zero. Si calcolano 4 valori e si negano. */
static void buildB(void)
{
    int p;
    for (p = 0; p < 64; p++) {
        int xa = XIN[2 * p];
        int xb = XIN[2 * p + 1];
        int s  = xa + xb;      /* voce 10, e -s e' la voce 0 */
        int d  = xa - xb;      /* voce  2, e -d e' la voce 8 */
        int base = (p >> 4) * 256 + (p & 15) * 16;

        PL[base + 10] = s;  PH[base + 10] = s >> 8;
        s = -s;
        PL[base]      = s;  PH[base]      = s >> 8;

        PL[base + 2]  = d;  PH[base + 2]  = d >> 8;
        d = -d;
        PL[base + 8]  = d;  PH[base + 8]  = d >> 8;

        PL[base + 6]  = xa; PH[base + 6]  = xa >> 8;
        xa = -xa;
        PL[base + 4]  = xa; PH[base + 4]  = xa >> 8;

        PL[base + 9]  = xb; PH[base + 9]  = xb >> 8;
        xb = -xb;
        PL[base + 1]  = xb; PH[base + 1]  = xb >> 8;

        PL[base + 5]  = 0;  PH[base + 5]  = 0;
    }
}

/* ---------- C: come B, ma con puntatori che avanzano ---------- */
static void buildC(void)
{
    int p;
    unsigned char *pl = PL;
    unsigned char *ph = PH;
    const signed char *x = XIN;

    for (p = 0; p < 64; p++) {
        int xa = x[0];
        int xb = x[1];
        int s  = xa + xb;
        int d  = xa - xb;

        pl[10] = s;  ph[10] = s >> 8;
        s = -s;
        pl[0]  = s;  ph[0]  = s >> 8;
        pl[2]  = d;  ph[2]  = d >> 8;
        d = -d;
        pl[8]  = d;  ph[8]  = d >> 8;
        pl[6]  = xa; ph[6]  = xa >> 8;
        xa = -xa;
        pl[4]  = xa; ph[4]  = xa >> 8;
        pl[9]  = xb; ph[9]  = xb >> 8;
        xb = -xb;
        pl[1]  = xb; ph[1]  = xb >> 8;
        pl[5]  = 0;  ph[5]  = 0;

        x  += 2;
        pl += 16;
        ph += 16;
    }
}

int main(void)
{
    int i, diffB, diffC;
    unsigned long cA, cB, cC;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"build tabelle: 3 varianti"); nl(); nl();

    /* A, e ne conserva il risultato come riferimento */
    tmr_start(); buildA(); cA = tmr_stop();
    for (i = 0; i < 1024; i++) { QL[i] = PL[i]; QH[i] = PH[i]; }

    for (i = 0; i < 1024; i++) { PL[i] = 0; PH[i] = 0; }
    tmr_start(); buildB(); cB = tmr_stop();
    diffB = 0;
    for (i = 0; i < 1024; i++) if (PL[i] != QL[i] || PH[i] != QH[i]) diffB++;

    for (i = 0; i < 1024; i++) { PL[i] = 0; PH[i] = 0; }
    tmr_start(); buildC(); cC = tmr_stop();
    diffC = 0;
    for (i = 0; i < 1024; i++) if (PL[i] != QL[i] || PH[i] != QH[i]) diffC++;

    put(p"A attuale  : "); putlong(cA); nl();
    put(p"B simmetria: "); putlong(cB);
    put(p"  diff "); putlong((unsigned long)diffB); nl();
    put(p"C puntatori: "); putlong(cC);
    put(p"  diff "); putlong((unsigned long)diffC); nl(); nl();

    put(p"per token x112:"); nl();
    put(p"  A "); putlong(cA * 112UL); nl();
    put(p"  C "); putlong(cC * 112UL); nl(); nl();

    if (diffB == 0 && diffC == 0) put(p"TUTTE IDENTICHE");
    else                          put(p"DIFFERENZE - NON USARE");

    while (true)
        ;
    return 0;
}