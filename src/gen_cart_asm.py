"""gen_cart_asm.py — genera cart4.c: kernel MAC in assembly 6502 che legge i
pesi da un banco di cartuccia EasyFlash, cronometrato col timer CIA.

BUG TROVATO E CORRETTO — collisione in pagina zero
--------------------------------------------------
Con -O3 e' attivo -Oz, che piazza automaticamente le variabili globali in
pagina zero. Il listato .asm lo mostrava senza ambiguita':

    scarto:  00f7
    giri:    00f8
    rowres:  00fa      <-- occupa $FA e $FB

Il mio assembly usava $FB/$FC come puntatore ai pesi: sovrapposizione su $FB.
Il compilatore stesso annotava  lda ($fb),y ; (rowres + 1)  e alla fine della
funzione scriveva il risultato sopra il puntatore:

    STA $fa  ; (rowres + 0)
    STA $fb  ; (rowres + 1)   <-- distrugge il puntatore
    ADC #$20
    STA $fb                    <-- avanza rowres, non il puntatore

Ecco perche' $FB finiva sempre a $8020: veniva ricostruito da zero a ogni
chiamata, e tutte le righe leggevano gli stessi pesi.

Rimedi:
  - puntatore spostato su $F4/$F5
  - compilare con -O2, non -O3, per non riattivare -Oz

Lezione: quando si mescola assembly inline e ottimizzatore, il file .asm
generato e' l'unico posto dove si vede davvero cosa succede.

Riferimenti: v5 in C da cartuccia 463.322 cicli (28/MAC)
"""

COLONNE = 128
RIGHE = 128
BYTE_PER_RIGA = COLONNE // 4
BLOCCHI = (COLONNE // 2 + 15) // 16

# pagina zero usata dall'assembly: scelta lontana da quella del compilatore
PTR_LO, PTR_HI = "$f4", "$f5"      # puntatore alla riga dei pesi
ACC_LO, ACC_HI = "$f2", "$f3"      # accumulatore a 16 bit
TMP = "$02"                        # temporaneo

print(f"{COLONNE} colonne -> {BYTE_PER_RIGA} byte/riga, {BLOCCHI} blocchi, "
      f"tabelle {BLOCCHI * 256} byte ciascuna")
print(f"pagina zero: puntatore {PTR_LO}/{PTR_HI}, accumulatore {ACC_LO}/{ACC_HI}")


def blocco(j):
    p_lo, p_hi = 2 * j, 2 * j + 1
    base = (p_lo >> 4) * 256
    m_lo = (p_lo & 15) * 16
    m_hi = (p_hi & 15) * 16
    off = f" + {base}" if base else ""

    a = [f"        ldy #{j}",
         f"        lda ({PTR_LO}),y",
         f"        sta {TMP}",
         "        and #15"]
    if m_lo:
        a.append(f"        ora #{m_lo}")
    a += ["        tay", "        clc",
          f"        lda {ACC_LO}", f"        adc PL{off},y", f"        sta {ACC_LO}",
          f"        lda {ACC_HI}", f"        adc PH{off},y", f"        sta {ACC_HI}",
          f"        lda {TMP}",
          "        lsr", "        lsr", "        lsr", "        lsr"]
    if m_hi:
        a.append(f"        ora #{m_hi}")
    a += ["        tay", "        clc",
          f"        lda {ACC_LO}", f"        adc PL{off},y", f"        sta {ACC_LO}",
          f"        lda {ACC_HI}", f"        adc PH{off},y", f"        sta {ACC_HI}"]
    return a


asm = [f"        lda #0", f"        sta {ACC_LO}", f"        sta {ACC_HI}"]
for j in range(BYTE_PER_RIGA):
    asm.append(f"        // byte {j}")
    asm += blocco(j)
asm += [f"        lda {ACC_LO}", "        sta rowres",
        f"        lda {ACC_HI}", "        sta rowres + 1",
        f"        // avanza di {BYTE_PER_RIGA} byte",
        "        clc", f"        lda {PTR_LO}", f"        adc #{BYTE_PER_RIGA}",
        f"        sta {PTR_LO}", "        bcc avanti", f"        inc {PTR_HI}",
        "avanti:"]

corpo = "\n".join(asm)

sorgente = f'''/* cart4.c — GENERATO DA gen_cart_asm.py, non modificare a mano.
   Kernel MAC in assembly 6502, pesi nel banco 1 della cartuccia,
   tempo misurato col timer B della CIA 2.

   ATTENZIONE: compilare con -O2, NON -O3.
   -O3 include -Oz, che piazza le globali in pagina zero e collide con
   la zona usata dall'assembly.

   Compilare:  oscar64 -O2 -tf=crt cart4.c
   Eseguire :  VICE, Attach cartridge image -> cart4.crt   (warp OFF!)

   Bersaglio: somma -17721, puntatore finale 36864.
   Riferimento: v5 in C da cartuccia 463.322 cicli (28/MAC) */

#include <c64/memmap.h>
#include <c64/charwin.h>
#include <c64/cia.h>
#include <c64/vic.h>
#include <c64/easyflash.h>
#include "c64data.h"

#pragma region( main, 0x0900, 0x8000, , , {{ code, data, bss, heap, stack }} )

#pragma section( bdata1, 0 )
#pragma region( bank1, 0x8000, 0xc000, , 1, {{ bdata1 }} )
#pragma data(bdata1)
__export const char WQBANK[{RIGHE * BYTE_PER_RIGA}] = {{
#embed {RIGHE * BYTE_PER_RIGA} 0 "wq.bin"
}};
#pragma data(data)

unsigned char PL[{BLOCCHI * 256}];
unsigned char PH[{BLOCCHI * 256}];
int rowres;
int RES[{RIGHE}];

static CharWin cw;
static volatile byte scarto;
static unsigned giri;

static void put(const char * s) {{ cwin_put_string(&cw, s, 7); }}
static void nl(void)            {{ cwin_cursor_newline(&cw); }}

static void putlong(unsigned long v)
{{
    char buf[12];
    char i = 12;
    if (v == 0) {{ cwin_put_char(&cw, p'0', 7); return; }}
    while (v) {{ buf[--i] = p'0' + (char)(v % 10); v /= 10; }}
    while (i < 12) cwin_put_char(&cw, buf[i++], 7);
}}

static void putnum(int v)
{{
    if (v < 0) {{ cwin_put_char(&cw, p'-', 7); putlong((unsigned long)(-v)); }}
    else       putlong((unsigned long)v);
}}

static void tmr_start(void)
{{
    cia2.crb = 0x00;
    cia2.tb  = 0xffff;
    scarto = cia2.icr;
    giri = 0;
    cia2.crb = 0x11;
}}

static void tmr_poll(void)
{{
    if (cia2.icr & 0x02) giri++;
}}

static unsigned long tmr_stop(void)
{{
    unsigned resto;
    cia2.crb = 0x00;
    tmr_poll();
    resto = 0xffff - cia2.tb;
    return (unsigned long)giri * 65536UL + resto;
}}

/* Una riga del prodotto matrice-vettore.
   Il puntatore ai pesi vive in {PTR_LO}/{PTR_HI} e avanza da solo. */
__noinline void macrow(void)
{{
    __asm
    {{
{corpo}
    }}
}}

int main(void)
{{
    int i, p, v, base, somma;
    unsigned long c_build, c_mac;

    mmap_set(MMAP_ROM);
    cia_init();
    vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1800);
    cwin_init(&cw, (char *)0x0400, 0, 0, 40, 25);
    cwin_clear(&cw);

    put(p"kernel asm da cartuccia"); nl(); nl();

    tmr_start();
    for (p = 0; p < {COLONNE // 2}; p++) {{
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
        tmr_poll();
    }}
    c_build = tmr_stop();

    put(p"calcolo..."); nl();

    eflash.bank = 1;
    tmr_start();

    *(const char **)0x00F4 = WQBANK;
    for (i = 0; i < {RIGHE}; i++) {{
        macrow();
        RES[i] = rowres;
        tmr_poll();
    }}

    c_mac = tmr_stop();

    somma = 0;
    for (i = 0; i < {RIGHE}; i++) somma += RES[i];

    put(p"ptr dopo: "); putlong(*(unsigned *)0x00F4); nl();
    put(p"atteso  : "); putlong((unsigned)WQBANK + {RIGHE * BYTE_PER_RIGA}); nl();

    put(p"primi 8:"); nl();
    for (i = 0; i < 8; i++) {{ putnum(RES[i]); cwin_put_char(&cw, p' ', 7); }}
    nl(); nl();

    put(p"somma : "); putnum(somma);
    put(p"  atteso -17721"); nl(); nl();

    put(p"cicli build : "); putlong(c_build); nl();
    put(p"cicli mac   : "); putlong(c_mac); nl();
    put(p"per mac     : "); putlong(c_mac / {RIGHE * COLONNE}UL); nl();
    put(p"in C era    : 463322 (28/mac)"); nl(); nl();

    if (somma == -17721) put(p"CORRETTO");
    else                 put(p"DIVERSO");

    while (true)
        ;
    return 0;
}}
'''

open("cart4.c", "w").write(sorgente)
n = corpo.count("\n") + 1
print(f"cart4.c scritto: {n} righe di assembly")
print("COMPILARE CON -O2, NON -O3")