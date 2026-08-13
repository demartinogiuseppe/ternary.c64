"""gen_macq.py - writes macq.c: the 16x8 MAC in assembly, AS A COMPONENT
ON ITS OWN, so it can be verified and timed before being grafted anywhere.

It computes Score(t) = sum_d Q[d]*K[t][d] with Q int16 and K int8: the
scoring loop of the attention. 512 scores, 0 errors, 81 cycles per
multiply-accumulate against 560 for the same thing in C.

Building it separately was worth it twice over: the number told us whether
grafting it was worth the work, and later, when a mechanical edit paired the
low table with the high one, the component caught it in five seconds. In the
big program that same mistake would have been one wrong number among
seventeen stages.

Note: inline comments below are in Italian.

--- appunti originali -------------------------------------------------


Serve a sostituire la moltiplicazione dell'attenzione, che in C costa 560
cicli per MAC contro i 21 del kernel ternario (sec. 9-bis).  Qui non si
innesta niente: si costruisce, si verifica contro un prodotto scalare in C e
si cronometra.  Se il numero non e' buono, non vale la pena di innestarlo.

COSA CALCOLA
    Score(t) = somma_d  Q[d] * K[t][d]        d = 0..15,  t = 0..NT-1
con Q int16 e K int8.  E' l'anello dei punteggi dell'attenzione: un vettore
Q fisso per testa contro NT righe di cache.

COME
1. Moltiplicazione a quarti di quadrato, solo SENZA SEGNO:
       a*b = f(a+b) - f(a-b)        f(x) = x*x/4
   quattro tabelle da 512 byte allineate alla pagina; l'indice a 9 bit e'
   (base + a) + b, con a nel puntatore e b nell'indice.  ~21 cicli.
   ⚠️ Le tabelle DEVONO essere allineate: #pragma align(nome, 256).

2. Tutti i prodotti diventano senza segno spostando gli operandi, e la
   correzione si applica UNA VOLTA per prodotto scalare, non per elemento:

     cache memorizzata gia' spostata:  ku[d][t] = k[t][d] + 128,  in [0,255]
     q spezzato:  q = qh*256 + ql,  qh = q>>8 con segno,  ql = q&255
     qh spostato: qhu = qh + 128,  in [0,255]

     Score(t) = 256 * ( somma_d qhu[d]*ku[d][t]  -  128*Ksum[t] )
                     +   somma_d  ql[d]*ku[d][t]
                     -   128*Sq

   con Ksum[t] = somma_d ku[d][t], che si calcola quando si SCRIVE la riga di
   cache, e Sq = somma_d q[d], una volta per testa.  Nessuna moltiplicazione
   con segno, nessuna correzione dentro l'anello.

3. ⚠️ LA CACHE K VA TRASPOSTA a [d][t].  Cosi' l'anello interno scorre t con
   il vettore Q fisso: l'operando a 16 bit resta costante per tutto l'anello
   e i suoi otto puntatori si preparano una volta per d, non per elemento.
   La scrittura diventa sparsa - 16 byte sparpagliati per posizione - contro
   16 letture contigue per testa: si paga una volta e si guadagna otto.

4. Due accumulatori a 24 bit PER POSIZIONE, tenuti come tre array di byte
   paralleli (ACH0/1/2, ACL0/1/2) cosi' si indicizzano con X senza
   moltiplicare per tre.  Massimo 16*65025 = 1.040.400: tre byte bastano.

PAGINA ZERO: $02-$11 gli otto puntatori, $12/$13 il prodotto.  Fuori dalla
zona del compilatore (regione zeropage a $F7) e da quella del kernel
ternario ($F2-$F5).

Uso:  python gen_macq.py
      .\\c64.ps1 macq -Cicli 40000000
"""

ND = 16          # dimensioni per testa
NT = 64          # posizioni: SEQ del programma vero
NPROVE = 8       # quanti vettori Q diversi si verificano

# ---------------------------------------------------------- tabelle f(x)=x^2/4
T1 = [(x * x) // 4 for x in range(512)]
T2 = [((x - 255) * (x - 255)) // 4 for x in range(512)]
assert max(T1) < 65536 and max(T2) < 65536


def byt(v, alto):
    return [(x >> 8) & 255 if alto else x & 255 for x in v]


def lista(v):
    return ",".join(str(x) for x in v)


# ---------------------------------------------------------------- l'assembly
# ⚠️⚠️ PAGINA ZERO: NON $02-$13.  WORK di oscar64 occupa $03-$0B e ci tiene
# valori vivi ATTRAVERSO una chiamata di funzione: un assembly inline che ci
# scrive dentro pianta il programma senza un messaggio.  Qui a se' non si
# vedeva - fra il calcolo dei puntatori e l'uso non c'era niente del
# compilatore da distruggere - e si e' visto solo innestandolo (sec. 11-ter).
# Un componente verificato a se' NON e' verificato nel contesto.
PZ_PTR = 0xE0      # 16 byte: gli otto puntatori
PZ_PLO = 0xF0
PZ_PHI = 0xF1


def blocco(d):
    """un d: prepara gli otto puntatori, poi scorre tutte le posizioni"""
    b = ["        // ---- d = %d ----" % d]
    for k in range(8):
        b += ["        lda PTRS + %d" % (d * 16 + 2 * k),
              "        sta $%02x" % (PZ_PTR + 2 * k),
              "        lda PTRS + %d" % (d * 16 + 2 * k + 1),
              "        sta $%02x" % (PZ_PTR + 1 + 2 * k)]
    b += ["        ldx #0",
          "lp%d:" % d,
          "        lda KU + %d,x" % (d * NT),
          "        tay"]
    # prodotto alto: qhu[d] * ku  ->  ACH
    b += ["        sec",
          "        lda ($%02x),y" % PZ_PTR, "        sbc ($%02x),y" % (PZ_PTR + 2), "        sta $%02x" % PZ_PLO,
          "        lda ($%02x),y" % (PZ_PTR + 4), "        sbc ($%02x),y" % (PZ_PTR + 6), "        sta $%02x" % PZ_PHI,
          "        clc",
          "        lda ACH0,x", "        adc $%02x" % PZ_PLO, "        sta ACH0,x",
          "        lda ACH1,x", "        adc $%02x" % PZ_PHI, "        sta ACH1,x",
          "        bcc nh%d" % d,
          "        inc ACH2,x",
          "nh%d:" % d]
    # prodotto basso: ql[d] * ku  ->  ACL
    b += ["        sec",
          "        lda ($%02x),y" % (PZ_PTR + 8), "        sbc ($%02x),y" % (PZ_PTR + 10), "        sta $%02x" % PZ_PLO,
          "        lda ($%02x),y" % (PZ_PTR + 12), "        sbc ($%02x),y" % (PZ_PTR + 14), "        sta $%02x" % PZ_PHI,
          "        clc",
          "        lda ACL0,x", "        adc $%02x" % PZ_PLO, "        sta ACL0,x",
          "        lda ACL1,x", "        adc $%02x" % PZ_PHI, "        sta ACL1,x",
          "        bcc nl%d" % d,
          "        inc ACL2,x",
          "nl%d:" % d]
    b += ["        inx",
          "        cpx NTC",
          "        bne lp%d" % d]
    return b


asm = []
for d in range(ND):
    asm += blocco(d)
CORPO = "\n".join(asm)

sorgente = r'''/* macq.c - GENERATO DA gen_macq.py.  MAC 16x8 in assembly, componente a se'.

   Score(t) = somma_d Q[d] * K[t][d],  Q int16, K int8, d = 0..15.
   E' l'anello dei punteggi dell'attenzione.  Qui si verifica contro un
   prodotto scalare in C e si cronometra: l'innesto viene dopo.

   Bersaglio da battere: 560 cicli per MAC, che e' quanto costa lo stesso
   calcolo in C con mul32by8 (sec. 9-bis).

   ⚠️ COMPILARE CON -O2, NON -O3.
   Eseguire:  .\c64.ps1 macq -Cicli 40000000
*/

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>

#pragma region( main, 0x0900, 0x8000, , , { code, data, bss, stack } )
#pragma stacksize( 2048 )

#define ND  __ND__
#define NT  __NT__

/* ⚠️ ALLINEATE ALLA PAGINA: l'indirizzo e' (tabella + a) + b con a nel
   puntatore e b nell'indice, quindi il byte basso della tabella deve essere
   zero e la tabella deve coprire 512 byte.
   ⚠️ #pragma align va DOPO la dichiarazione, non prima: prima da'
   "error 3005: Variable not found". */
const unsigned char T1L[512] = {__T1L__};
#pragma align( T1L, 256 )
const unsigned char T1H[512] = {__T1H__};
#pragma align( T1H, 256 )
const unsigned char T2L[512] = {__T2L__};
#pragma align( T2L, 256 )
const unsigned char T2H[512] = {__T2H__};
#pragma align( T2H, 256 )

unsigned char KU[ND][NT];        /* cache TRASPOSTA e gia' spostata di +128 */
unsigned       KSUM[NT];         /* somma_d ku[d][t], al momento di scrivere */
int            Q[ND];
unsigned       PTRS[ND][8];      /* gli otto puntatori per ogni d */
unsigned char  NTC;              /* quante posizioni davvero */

unsigned char ACH0[NT], ACH1[NT], ACH2[NT];
unsigned char ACL0[NT], ACL1[NT], ACL2[NT];

long SC[NT], REF[NT];

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

static void tmr_poll(void) { if (cia2.icr & 0x02) giri++; }

static unsigned long tmr_stop(void)
{
    unsigned resto;
    cia2.crb = 0x00;
    tmr_poll();
    resto = 0xffff - cia2.tb;
    return (unsigned long)giri * 65536UL + resto;
}

/* Prepara gli otto puntatori di ogni d.  Una volta per testa, non per
   elemento: e' tutto il senso della trasposizione della cache. */
static void prep(void)
{
    int d, qq;
    unsigned char qhu, ql;
    for (d = 0; d < ND; d++) {
        qq  = Q[d];
        qhu = (unsigned char)((qq >> 8) + 128);
        ql  = (unsigned char)(qq & 255);
        PTRS[d][0] = (unsigned)T1L + qhu;
        PTRS[d][1] = (unsigned)T2L + (unsigned char)(qhu ^ 255);
        PTRS[d][2] = (unsigned)T1H + qhu;
        PTRS[d][3] = (unsigned)T2H + (unsigned char)(qhu ^ 255);
        PTRS[d][4] = (unsigned)T1L + ql;
        PTRS[d][5] = (unsigned)T2L + (unsigned char)(ql ^ 255);
        PTRS[d][6] = (unsigned)T1H + ql;
        PTRS[d][7] = (unsigned)T2H + (unsigned char)(ql ^ 255);
    }
}

__noinline void macq(void)
{
    __asm
    {
__CORPO__
    }
}

int main(void)
{
    int d, t, prova;
    long s, sq;
    unsigned long c_mac = 0, c_comb = 0, cc;
    unsigned semek, semeq;
    int errori = 0;
    int pr_t = -1;
    long pr_ott = 0, pr_att = 0;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"mac 16x8, "); putnum((long)ND); put(p" dim x ");
    putnum((long)NT); put(p" pos"); nl();

    NTC = NT;
    semek = 31337;
    semeq = 12345;

    for (prova = 0; prova < __NPROVE__; prova++) {

        /* cache: valori int8 sparsi su tutto l'intervallo, memorizzati
           spostati di +128, e la somma di riga calcolata scrivendo */
        for (t = 0; t < NT; t++) KSUM[t] = 0;
        for (d = 0; d < ND; d++) {
            for (t = 0; t < NT; t++) {
                semek = semek * 25173 + 13849;
                KU[d][t] = (unsigned char)(semek >> 8);
                KSUM[t] += (unsigned)KU[d][t];
            }
        }
        /* Q su tutto l'intervallo di un int16, estremi compresi */
        for (d = 0; d < ND; d++) {
            semeq = semeq * 25173 + 13849;
            Q[d] = (int)semeq;
        }
        if (prova == 0) { Q[0] = 32767; Q[1] = -32768; Q[2] = 0; Q[3] = -1; }

        sq = 0;
        for (d = 0; d < ND; d++) sq += (long)Q[d];
        sq = sq << 7;

        /* riferimento in C, la strada lunga */
        for (t = 0; t < NT; t++) {
            s = 0;
            for (d = 0; d < ND; d++)
                s += (long)Q[d] * (long)((int)KU[d][t] - 128);
            REF[t] = s;
        }

        for (t = 0; t < NT; t++) {
            ACH0[t] = 0; ACH1[t] = 0; ACH2[t] = 0;
            ACL0[t] = 0; ACL1[t] = 0; ACL2[t] = 0;
        }
        prep();

        tmr_start();
        macq();
        c_mac += tmr_stop();

        tmr_start();
        for (t = 0; t < NT; t++) {
            long ach = (long)ACH0[t]
                     + ((long)ACH1[t] << 8)
                     + ((long)ACH2[t] << 16);
            long acl = (long)ACL0[t]
                     + ((long)ACL1[t] << 8)
                     + ((long)ACL2[t] << 16);
            ach = ach - 128L * (long)KSUM[t];
            s = (ach << 8) + acl - sq;
            SC[t] = s;
            tmr_poll();
        }
        c_comb += tmr_stop();

        for (t = 0; t < NT; t++) {
            if (SC[t] != REF[t]) {
                errori++;
                if (pr_t < 0) { pr_t = t; pr_ott = SC[t]; pr_att = REF[t]; }
            }
        }
    }

    putnum((long)(__NPROVE__ * NT)); put(p" punteggi, ");
    putnum((long)errori); put(p" errori"); nl();
    if (errori == 0) put(p"MAC ASM OK");
    else {
        put(p"primo t"); putnum((long)pr_t);
        put(p" "); putnum(pr_ott);
        put(p" != "); putnum(pr_att);
    }
    nl(); nl();

    c_mac  /= __NPROVE__;
    c_comb /= __NPROVE__;
    cc = c_mac + c_comb;

    put(p"asm      : "); putnum((long)c_mac);   nl();
    put(p"combina  : "); putnum((long)c_comb);  nl();
    put(p"totale   : "); putnum((long)cc);      nl();
    put(p"per mac  : "); putnum((long)(cc / ((unsigned long)ND * NT)));
    put(p"   in C era 560"); nl();
    put(p"solo asm : "); putnum((long)(c_mac / ((unsigned long)ND * NT)));
    nl();

    while (true)
        ;
    return 0;
}
'''

sorgente = (sorgente
            .replace("__ND__", str(ND))
            .replace("__NT__", str(NT))
            .replace("__NPROVE__", str(NPROVE))
            .replace("__T1L__", lista(byt(T1, False)))
            .replace("__T1H__", lista(byt(T1, True)))
            .replace("__T2L__", lista(byt(T2, False)))
            .replace("__T2H__", lista(byt(T2, True)))
            .replace("__CORPO__", CORPO))
open("macq.c", "w", encoding="utf-8").write(sorgente)

print("macq.c scritto: %d dimensioni x %d posizioni, %d righe di assembly"
      % (ND, NT, CORPO.count("\n") + 1))
print("il corpo dell'anello interno e' %d istruzioni: il bne all'indietro "
      "deve restare entro 127 byte" % (len(blocco(0)) - 16))
print("COMPILARE CON -O2, NON -O3")
