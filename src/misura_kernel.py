"""misura_kernel.py — quanto codice occupa davvero un kernel srotolato?

La stima di 2,6 KB per il kernel a 32 byte per riga era estrapolata. Il file
.map di oscar64 dà il dato vero, nel formato:

    088e (0654) : macrow, NATIVE_CODE:code

dove 0654 esadecimale = 1620 byte, la dimensione della funzione.

Genera tre sorgenti minimi (4, 32, 88 byte per riga), li compila, e legge la
dimensione reale. Nessun emulatore: solo tre compilazioni.

Vincolo da verificare: in un banco EasyFlash da 16384 byte devono stare
INSIEME il kernel e i pesi della matrice che legge (codice e dati condividono
la finestra a $8000).

Uso:  python misura_kernel.py
"""

import os
import re
import subprocess

OSCAR = r"C:\oscar64\bin\oscar64.exe"
BANCO = 16384

LARGHEZZE = [
    (16,  "wk, wv",         16),    # 16 righe
    (128, "wq, wo",         128),
    (128, "w1, w3 (352 r.)", 352),
    (352, "w2",             128),
]


def blocco(j):
    p_lo, p_hi = 2 * j, 2 * j + 1
    base = (p_lo >> 4) * 256
    m_lo = (p_lo & 15) * 16
    m_hi = (p_hi & 15) * 16
    off = f" + {base}" if base else ""
    a = [f"        ldy #{j}", "        lda ($f4),y", "        sta $02",
         "        and #15"]
    if m_lo:
        a.append(f"        ora #{m_lo}")
    a += ["        tay", "        clc",
          "        lda $f2", f"        adc PL{off},y", "        sta $f2",
          "        lda $f3", f"        adc PH{off},y", "        sta $f3",
          "        lda $02", "        lsr", "        lsr", "        lsr",
          "        lsr"]
    if m_hi:
        a.append(f"        ora #{m_hi}")
    a += ["        tay", "        clc",
          "        lda $f2", f"        adc PL{off},y", "        sta $f2",
          "        lda $f3", f"        adc PH{off},y", "        sta $f3"]
    return a


def genera(colonne):
    bpr = colonne // 4
    blocchi = (colonne // 2 + 15) // 16
    asm = ["        lda #0", "        sta $f2", "        sta $f3"]
    for j in range(bpr):
        asm += blocco(j)
    asm += ["        lda $f2", "        sta rowres",
            "        lda $f3", "        sta rowres + 1",
            "        clc", "        lda $f4", f"        adc #{bpr}",
            "        sta $f4", "        bcc avanti", "        inc $f5",
            "avanti:"]
    corpo = "\n".join(asm)
    return f'''#include <c64/charwin.h>
unsigned char PL[{blocchi * 256}];
unsigned char PH[{blocchi * 256}];
int rowres;
__noinline void macrow(void)
{{
    __asm
    {{
{corpo}
    }}
}}
int main(void) {{ macrow(); return rowres; }}
''', bpr, blocchi


def dimensione(colonne):
    """compila un kernel e restituisce la dimensione in byte, o None"""
    src, bpr, blocchi = genera(colonne)
    nome = f"_km{colonne}.c"
    open(nome, "w").write(src)
    subprocess.run([OSCAR, "-O2", nome], capture_output=True)

    mapfile = nome.replace(".c", ".map")
    if not os.path.exists(mapfile):
        return None, bpr, blocchi
    for riga in open(mapfile):
        m = re.search(r"^[0-9a-f]{4}\s*\(([0-9a-f]{4})\)\s*:\s*macrow,", riga)
        if m:
            return int(m.group(1), 16), bpr, blocchi
    return None, bpr, blocchi


cache = {}
print(f"{'col':>5} {'matrici':<16} {'b/riga':>7} {'codice':>8} {'righe':>6} "
      f"{'pesi':>8} {'cod+pesi':>9} {'in 16K?':>8}")
print("-" * 76)

for col, nomi, righe in LARGHEZZE:
    if col not in cache:
        cache[col] = dimensione(col)
    dim, bpr, blocchi = cache[col]
    pesi = righe * bpr
    if dim is None:
        print(f"{col:>5} {nomi:<16} {bpr:>7} {'n.d.':>8}")
        continue
    tot = dim + pesi
    ok = "SI" if tot <= BANCO else "NO"
    print(f"{col:>5} {nomi:<16} {bpr:>7} {dim:>7} B {righe:>6} "
          f"{pesi:>6} B {tot:>7} B {ok:>8}")

print()
print("Le tabelle PL/PH stanno in RAM, non nel banco: non contano qui.")
print(f"Banco EasyFlash: {BANCO} byte.")