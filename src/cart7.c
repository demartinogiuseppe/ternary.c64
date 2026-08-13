/* cart7.c — moltiplicazione 8x8 con i quarti di quadrato.

   a*b = ( (a+b)^2 - (a-b)^2 ) / 4,  con f(n) = floor(n^2/4) tabellato.
   La parte frazionaria si cancella nella sottrazione: risultato esatto.

   Serve per i prodotti scalari dell'attenzione (q.k) e per RoPE, le uniche
   parti non ternarie del modello.

   Tre versioni misurate:
     A  C diretto con a*b      (quanto costa la moltiplicazione di oscar64)
     B  C con le tabelle
     C  assembly con le tabelle

   Tutte e tre devono dare gli stessi risultati sui 12 casi di prova, e
   B e C devono coincidere con A su un giro completo di 256x256.

   Compilare:  oscar64 -O2 -tf=crt -o=cart7.crt cart7.c
   (Detach in VICE PRIMA di ricompilare) */

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include "muldata.h"

#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, heap, stack } )

static CharWin cw;
static volatile byte scarto;
static unsigned giri;

/* argomenti e risultato per la versione assembly, in pagina zero fissa */
unsigned char MA, MB;
unsigned MR;

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
    cia2.crb = 0x00;  cia2.tb = 0xffff;
    scarto = cia2.icr;  giri = 0;
    cia2.crb = 0x11;
}
static void tmr_poll(void) { if (cia2.icr & 0x02) giri++; }
static unsigned long tmr_stop(void)
{
    unsigned resto;
    cia2.crb = 0x00;  tmr_poll();
    resto = 0xffff - cia2.tb;
    return (unsigned long)giri * 65536UL + resto;
}

/* ---------- B: tabelle in C ---------- */
static unsigned mulB(unsigned char a, unsigned char b)
{
    unsigned s = (unsigned)a + b;
    unsigned char d = (a > b) ? (a - b) : (b - a);
    unsigned hi = ((unsigned)SQL[s]) | ((unsigned)SQH[s] << 8);
    unsigned lo = ((unsigned)SQL[d]) | ((unsigned)SQH[d] << 8);
    return hi - lo;
}

/* ---------- C: tabelle in assembly ----------
   MA, MB ingressi; MR risultato.
   Usa $F0/$F1 come temporaneo, lontano dalla zona del compilatore. */
__noinline void mulC(void)
{
    __asm
    {
        // d = |MA - MB|, in X
        lda MA
        sec
        sbc MB
        bcs pos
        lda MB
        sec
        sbc MA
    pos:
        tax

        // s = MA + MB, puo' superare 255: byte alto in $f1
        lda #0
        sta $f1
        clc
        lda MA
        adc MB
        sta $f0
        bcc nocarry
        inc $f1
    nocarry:

        // MR = SQ[s] - SQ[d]
        // il byte alto di s vale 0 o 1: due rami, indirizzo base diverso
        lda $f1
        bne alta

        ldy $f0
        sec
        lda SQL,y
        sbc SQL,x
        sta MR
        lda SQH,y
        sbc SQH,x
        sta MR + 1
        rts

    alta:
        ldy $f0
        sec
        lda SQL + 256,y
        sbc SQL,x
        sta MR
        lda SQH + 256,y
        sbc SQH,x
        sta MR + 1
    }
}

int main(void)
{
    int i, errB, errC, errFull;
    unsigned a, b, r;
    unsigned long cA, cB, cC;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"moltiplicazione 8x8"); nl(); nl();

    /* correttezza sui casi di prova */
    errB = errC = 0;
    for (i = 0; i < NPROVE; i++) {
        if (mulB(PA[i], PB[i]) != PATT[i]) errB++;
        MA = PA[i]; MB = PB[i]; mulC();
        if (MR != PATT[i]) errC++;
    }
    put(p"prove: errB "); putlong((unsigned long)errB);
    put(p"  errC "); putlong((unsigned long)errC); nl();

    /* correttezza esaustiva di C contro la moltiplicazione nativa */
    put(p"giro completo..."); nl();
    errFull = 0;
    for (a = 0; a < 256; a++)
        for (b = 0; b < 256; b++) {
            MA = a; MB = b; mulC();
            if (MR != (unsigned)(a * b)) errFull++;
        }
    put(p"65536 coppie: "); putlong((unsigned long)errFull);
    put(p" errori"); nl(); nl();

    /* velocita': 4096 moltiplicazioni per versione */
    tmr_start();
    r = 0;
    for (a = 0; a < 64; a++)
        for (b = 0; b < 64; b++) r += (unsigned)(a * b);
    cA = tmr_stop();

    tmr_start();
    r = 0;
    for (a = 0; a < 64; a++)
        for (b = 0; b < 64; b++) r += mulB(a, b);
    cB = tmr_stop();

    tmr_start();
    for (a = 0; a < 64; a++)
        for (b = 0; b < 64; b++) { MA = a; MB = b; mulC(); }
    cC = tmr_stop();

    put(p"4096 moltiplicazioni:"); nl();
    put(p"A nativa : "); putlong(cA);
    put(p"  per mul "); putlong(cA / 4096UL); nl();
    put(p"B tabelle: "); putlong(cB);
    put(p"  per mul "); putlong(cB / 4096UL); nl();
    put(p"C asm    : "); putlong(cC);
    put(p"  per mul "); putlong(cC / 4096UL); nl(); nl();

    if (errB == 0 && errC == 0 && errFull == 0) put(p"CORRETTO");
    else                                        put(p"ERRORI");

    while (true)
        ;
    return 0;
}