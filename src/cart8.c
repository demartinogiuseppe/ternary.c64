/* cart8.c — CODICE in banchi diversi della cartuccia.

   Problema: il kernel srotola tutti i byte di una riga. Servono larghezze
   diverse (4, 32, 88 byte per riga) e quattro kernel distinti superano gli
   8 KB, su 28 KB totali di RAM. Soluzione: metterli in banchi di cartuccia
   e commutare prima di chiamarli.

   Questo test verifica il meccanismo su due kernel finti in due banchi,
   piu' i dati in un terzo. Se funziona, il resto e' generazione automatica.

   Da verificare:
    - una funzione in un banco viene eseguita correttamente
    - puo' leggere dati... ATTENZIONE: codice e dati sono nella STESSA finestra
      a $8000. Se il kernel sta nel banco 2 e i pesi nel banco 1, il kernel
      non puo' leggere i pesi mentre e' in esecuzione. Vanno nello STESSO banco.
    - il test lo dimostra: kernel e dati insieme nel banco, uno per larghezza

   Compilare:  oscar64 -O2 -tf=crt -o=cart8.crt cart8.c
   (Detach in VICE PRIMA di ricompilare) */

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>

#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, heap, stack } )

/* banco 1: kernel "stretto" (4 byte per riga) + i suoi dati */
#pragma section( bcode1, 0 )
#pragma section( bdata1, 0 )
#pragma region( bank1, 0x8000, 0xc000, , 1, { bcode1, bdata1 } )

/* banco 2: kernel "largo" (8 byte per riga) + i suoi dati */
#pragma section( bcode2, 0 )
#pragma section( bdata2, 0 )
#pragma region( bank2, 0x8000, 0xc000, , 2, { bcode2, bdata2 } )

/* risultato condiviso, in RAM: entrambi i banchi possono scriverci */
int risultato;
unsigned char TAB[16];

/* ---------------- banco 1 ---------------- */
#pragma code ( bcode1 )
#pragma data ( bdata1 )

__export const unsigned char DATI1[16] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
};

void somma_stretta(void)
{
    int i, s = 0;
    for (i = 0; i < 4; i++) s += DATI1[i] * TAB[i];
    risultato = s;
}

/* ---------------- banco 2 ---------------- */
#pragma code ( bcode2 )
#pragma data ( bdata2 )

__export const unsigned char DATI2[16] = {
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115
};

void somma_larga(void)
{
    int i, s = 0;
    for (i = 0; i < 8; i++) s += DATI2[i] * TAB[i];
    risultato = s;
}

/* ---------------- codice condiviso, in RAM ---------------- */
#pragma code ( code )
#pragma data ( data )

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

int main(void)
{
    int i, r1, r2, att1, att2;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"codice in banchi separati"); nl(); nl();

    for (i = 0; i < 16; i++) TAB[i] = i + 1;

    put(p"bank di somma_stretta: ");
    putnum((long)__bankof(somma_stretta)); nl();
    put(p"bank di somma_larga  : ");
    putnum((long)__bankof(somma_larga)); nl(); nl();

    /* atteso: 1*1 + 2*2 + 3*3 + 4*4 = 30 */
    att1 = 0;
    for (i = 0; i < 4; i++) att1 += (i + 1) * (i + 1);

    /* atteso: 100*1 + 101*2 + ... + 107*8 */
    att2 = 0;
    for (i = 0; i < 8; i++) att2 += (100 + i) * (i + 1);

    eflash.bank = __bankof(somma_stretta);
    somma_stretta();
    r1 = risultato;

    eflash.bank = __bankof(somma_larga);
    somma_larga();
    r2 = risultato;

    put(p"stretta: "); putnum((long)r1);
    put(p"  atteso "); putnum((long)att1); nl();
    put(p"larga  : "); putnum((long)r2);
    put(p"  atteso "); putnum((long)att2); nl(); nl();

    /* seconda tornata: verifica che la commutazione sia ripetibile */
    eflash.bank = __bankof(somma_stretta);
    somma_stretta();
    put(p"stretta di nuovo: "); putnum((long)risultato); nl(); nl();

    if (r1 == att1 && r2 == att2 && risultato == att1)
        put(p"BANCHI DI CODICE OK");
    else
        put(p"FALLITO");

    while (true)
        ;
    return 0;
}