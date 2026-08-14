"""monta_gif.py - monta la GIF a 35x con il cronometro sotto.

Gli scatti sono quelli veri di VICE, uno per parola (cattura_film.py).  Fra
una parola e l'altra lo schermo NON cambia - main() scrive solo quando il
token e' pronto - quindi 35 immagini bastano per tutti i fotogrammi, e
l'unica cosa che si muove e' il cronometro, che e' disegnato qui.

IL CRONOMETRO NON E' ANIMATO: i secondi escono dai cicli misurati dal
monitor di VICE, divisi per il clock PAL vero (985.248 Hz).  Il numero a
destra dice di quanto stai barando a guardarlo.

La barra e' scritta col font vero del C64, preso dalla chargen ROM di VICE,
negli stessi due colori che il programma mette nel VIC: sembra la
ventiseiesima riga della macchina invece di una didascalia incollata sopra.
"""

import json
import os
import sys
from collections import Counter

from PIL import Image

VELOCITA = 35            # quante volte piu' veloce del vero
FPS = 8                  # fotogrammi al secondo della GIF.  Non e' una scelta
                         # estetica: LinkedIn anima al massimo 400 fotogrammi e
                         # oltre quella soglia congela il primo.  A 10 fps erano
                         # 416.  Gli fps non toccano ne' la durata ne' VELOCITA:
                         # fotogrammi = durata x FPS, e qui si muove solo il
                         # cronometro, che avanza di ~4 secondi di C64 per
                         # fotogramma in entrambi i casi.
SCALA = 2                # lo schermo del C64 e' 384x272: troppo piccolo da telefono
DIR = "film"
USCITA = "film/ternary-c64-35x.gif"

# La chargen del C64.  4096 byte = due set da 256 caratteri; il programma
# chiama vic_setmode(..., 0x1800), cioe' il set misto maiuscole/minuscole,
# che e' il secondo.
CHARGEN = (r"G:\AAAiProject\Claude\c64\GTK3VICE-3.10-win64"
           r"\GTK3VICE-3.10-win64\C64\chargen-901225-01.bin")
SET_MISTO = 2048


def codice(ch):
    """ASCII -> codice schermo, nel set misto."""
    if "a" <= ch <= "z":
        return ord(ch) - ord("a") + 1
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 65
    if "0" <= ch <= "9":
        return ord(ch) - ord("0") + 48
    return {" ": 32, ":": 58, ".": 46, "-": 45, "/": 47}.get(ch, 32)


def glifi():
    d = os.environ.get("CHARGEN", CHARGEN)
    if not os.path.exists(d):
        sys.exit("chargen non trovata: %s\nimposta $env:CHARGEN" % d)
    rom = open(d, "rb").read()
    return rom[SET_MISTO:SET_MISTO + 2048]


def scrivi(img, rom, testo, x0, y0, colore, scala):
    """Disegna testo col font del C64.  Un pixel acceso = un quadrato scala x scala."""
    px = img.load()
    for i, ch in enumerate(testo):
        base = codice(ch) * 8
        for r in range(8):
            riga = rom[base + r]
            for c in range(8):
                if riga & (0x80 >> c):
                    x = x0 + (i * 8 + c) * scala
                    y = y0 + r * scala
                    for dy in range(scala):
                        for dx in range(scala):
                            px[x + dx, y + dy] = colore


def colori(scatto):
    """Bordo e sfondo presi DALLO SCATTO, non indovinati."""
    bordo = scatto.getpixel((2, 2))
    dentro = scatto.crop((32, 36, 352, 236))
    sfondo = Counter(dentro.getdata()).most_common(1)[0][0]
    return bordo, sfondo


def main():
    with open("%s/film.json" % DIR, encoding="utf-8") as f:
        dati = json.load(f)

    clock = dati["clock"]
    confini = dati["confini"]          # cicli a cui compare la parola k
    fine = dati["fine"]
    stati = dati["stati"]

    scatti = [Image.open(s["png"]).convert("RGB") for s in stati]
    if len(scatti) != len(confini) + 1:
        sys.exit("%d scatti per %d confini: non torna"
                 % (len(scatti), len(confini)))

    rom = glifi()
    bordo, sfondo = colori(scatti[0])

    L, A = scatti[0].size
    L, A = L * SCALA, A * SCALA
    ALTEZZA_BARRA = 16 * SCALA + 8 * SCALA        # una riga di testo, con aria

    # un fotogramma ogni 1/FPS di GIF = VELOCITA/FPS secondi di C64
    passo = clock * VELOCITA // FPS
    n = int(fine // passo) + 1

    print("%d stati, %.1f s di C64, %dx a %d fps -> %d fotogrammi, %.1f s di GIF"
          % (len(scatti), fine / clock, VELOCITA, FPS, n, n / FPS))

    fotogrammi = []
    k = 0
    for i in range(n):
        cicli = i * passo
        while k < len(confini) and confini[k] <= cicli:
            k += 1                      # k = quante parole sono gia' comparse

        f = Image.new("RGB", (L, A + ALTEZZA_BARRA), sfondo)
        f.paste(scatti[k].resize((L, A), Image.NEAREST), (0, 0))

        s = int(cicli / clock)
        riga = "C64 TIME %02d:%02d" % (s // 60, s % 60)
        riga += " " * (40 - len(riga) - 12) + "SHOWN AT %dx" % VELOCITA
        # allineata alla colonna 0 del testo sullo schermo (il bordo e' 32 px)
        scrivi(f, rom, riga, 32 * SCALA, A + 4 * SCALA, bordo, SCALA)
        fotogrammi.append(f)

    # una tavolozza sola per tutta la GIF: se ogni fotogramma si quantizza per
    # conto suo i colori ballano fra un fotogramma e l'altro
    madre = fotogrammi[0].convert("P", palette=Image.ADAPTIVE, colors=16)
    tavole = [f.quantize(palette=madre) for f in fotogrammi]

    durate = [1000 // FPS] * len(tavole)
    durate[-1] = 2000                   # l'ultima resta ferma: si legge la frase

    tavole[0].save(USCITA, save_all=True, append_images=tavole[1:],
                   duration=durate, loop=0, optimize=True, disposal=1)

    mb = os.path.getsize(USCITA) / 1048576
    print("%s scritto: %d fotogrammi, %.2f MB" % (USCITA, len(tavole), mb))
    print("ultima parola a %02d:%02d, fine a %02d:%02d"
          % (int(confini[-1] / clock) // 60, int(confini[-1] / clock) % 60,
             int(fine / clock) // 60, int(fine / clock) % 60))


if __name__ == "__main__":
    main()
