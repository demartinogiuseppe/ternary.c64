"""verifica_asm.py - controlla il listato .asm prodotto da oscar64.

Cerca le DIRAMAZIONI A SPOSTAMENTO ZERO: un branch condizionale che salta a
se stesso, cioe' non salta.  Sono il sintomo di un'etichetta di assembly
risolta su un simbolo sbagliato.

Da dove nasce.  Le routine __asm del runtime di oscar64 (mul32, mul32by8,
divmod32) usano etichette interne dai nomi cortissimi: W1 W2 W3 W4 WW L1 E0
W16 W32 LB1 LB1a LB1b WB1 LS0 LS1 LS1a LS1b WS1 zero.  Se il programma
dichiara una GLOBALE con uno di quei nomi, l'etichetta viene risolta sulla
globale.  Il salto punta all'array (in cartuccia, a $8000): fuori portata per
un branch relativo, e l'assemblatore emette spostamento ZERO senza alcun
messaggio.  Da quel momento mul32 e divmod32 restituiscono valori sbagliati,
in silenzio, e ogni riscalamento in virgola fissa e' spazzatura.

Ci sono volute ore per trovarlo perche' non c'e' nessun errore di
compilazione: il .asm e il .crt COINCIDONO, sono entrambi sbagliati.
Questo controllo costa un decimo di secondo. Va eseguito dopo OGNI build.

Uso:
    python verifica_asm.py programma.asm [altro.asm ...]

Restituisce 1 se trova qualcosa, cosi' si puo' incatenare alla compilazione.
"""

import re
import sys

BRANCH = {
    "10": "BPL", "30": "BMI", "50": "BVC", "70": "BVS",
    "90": "BCC", "b0": "BCS", "d0": "BNE", "f0": "BEQ",
}

# etichette interne del runtime: una globale con questi nomi rompe tutto
NOMI_VIETATI = {
    "W1", "W2", "W3", "W4", "WW", "L1", "E0", "W16", "W32",
    "LB1", "LB1A", "LB1B", "WB1", "LS0", "LS1", "LS1A", "LS1B", "WS1",
    "ZERO",
}

RE_ISTR = re.compile(r"^([0-9a-f]{4}) : ([0-9a-f]{2}) ([0-9a-f]{2}) __ (B[A-Z]{2})")
RE_ETIC = re.compile(r"^(\w+):")
# nella mappa i simboli di banco hanno il prefisso del banco (01:),
# quelli in RAM hanno --:
RE_GLOB = re.compile(r"^(?:--|[0-9a-f]{2}):[0-9a-f]{4} - [0-9a-f]{4} : (\w+), DATA:")


def controlla(percorso):
    try:
        righe = open(percorso, encoding="utf-8", errors="replace").read().splitlines()
    except OSError as e:
        print("  non leggibile: %s" % e)
        return 1

    funzione = "?"
    guasti = []
    totale = 0
    for riga in righe:
        m = RE_ISTR.match(riga)
        if m:
            indirizzo, opcode, spost, mnem = m.groups()
            if opcode in BRANCH:
                totale += 1
                if spost == "00":
                    guasti.append((indirizzo, mnem, funzione))
            continue
        m = RE_ETIC.match(riga)
        if m:
            funzione = m.group(1)

    if not guasti:
        print("  %-22s %3d diramazioni, tutte sane" % (percorso, totale))
        return 0

    print("  %-22s %3d diramazioni, %d A SPOSTAMENTO ZERO:"
          % (percorso, totale, len(guasti)))
    for indirizzo, mnem, funzione in guasti:
        print("      $%s  %s  dentro %s" % (indirizzo, mnem, funzione))
    print("      Cerca una globale che si chiami come un'etichetta del runtime:")
    print("      " + " ".join(sorted(NOMI_VIETATI)))
    return 1


def controlla_nomi(percorso):
    """segnala globali con nomi che collidono, anche prima del danno"""
    sospette = []
    try:
        mappa = percorso[:-4] + ".map"
        for riga in open(mappa, encoding="utf-8", errors="replace"):
            m = RE_GLOB.match(riga)
            if m and m.group(1).upper() in NOMI_VIETATI:
                sospette.append(m.group(1))
    except OSError:
        return 0
    if sospette:
        print("      globali dal nome pericoloso: %s" % ", ".join(sorted(set(sospette))))
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    esito = 0
    print("verifica dei listati:")
    for percorso in argv[1:]:
        esito |= controlla(percorso)
        controlla_nomi(percorso)
    if esito:
        print("ESITO: LISTATO GUASTO - non eseguire il .crt")
    else:
        print("ESITO: tutto sano")
    return esito


if __name__ == "__main__":
    sys.exit(main(sys.argv))
