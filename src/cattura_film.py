"""cattura_film.py - le schermate vere per la GIF, una per parola.

COME.  Il monitor di VICE si ferma a ogni chiamata di stampa_token ($127C),
legge il cronometro, salva uno snapshot della macchina ed esce; la sessione
dopo riprende da quello snapshot.  Costo LINEARE: ogni sessione emula un solo
token invece di rifare tutto da capo, quindi ~4 secondi di orologio ciascuna
invece dei ~200 dell'ultima.

⚠️ Alla k-esima chiamata il token k NON e' ancora stampato: sullo schermo ce
ne sono k-1.  Quindi lo scatto della sessione k e' lo stato con k-1 parole, e
il cronometro della sessione k e' l'istante in cui compare la parola k.

⚠️ -limitcycles si confronta con l'orologio ASSOLUTO della macchina, non con
i cicli di questa sessione: dopo un undump il contatore riparte da dov'era.
Un tetto fisso di 80 M avrebbe fatto abortire subito ogni sessione dalla
terza in poi.  Qui il tetto e' sempre "dove ero + margine".

⚠️ Il clock e' quello PAL vero, 985.248 Hz.  c64.ps1 usa il milione tondo per
dare un tetto ai run, ma su venti minuti la differenza e' diciotto secondi, e
questi numeri finiscono sotto gli occhi di chi guarda la GIF.
"""

import json
import os
import re
import subprocess
import sys

CLOCK = 985248                  # C64 PAL, Hz
STAMPA_TOKEN = "127c"           # da genfilm.lbl
NGEN = 34                       # token generati (gendata.h)
MARGINE = 120_000_000           # tetto di sicurezza per sessione, in cicli
CODA = 8_000_000                # cicli in piu' per l'ultimo scatto (il "done.")

DIR = "film"
CRT = "genfilm.crt"


def vice():
    v = os.environ.get("VICE")
    if v and os.path.exists(v):
        return v
    d = (r"G:\AAAiProject\Claude\c64\GTK3VICE-3.10-win64"
         r"\GTK3VICE-3.10-win64\bin\x64sc.exe")
    if os.path.exists(d):
        return d
    sys.exit("VICE non trovato: imposta $env:VICE al percorso di x64sc.exe")


def sessione(nome, comandi, tetto, png):
    """Una corsa di VICE guidata dal monitor.  Torna il cronometro, o None.

    ⚠️ TUTTI i percorsi vanno con la barra dritta.  Dentro le virgolette dei
    comandi del monitor la barra rovesciata e' un escape e il dump finisce nel
    nulla senza dire niente; passata a -monlogname non produce nemmeno il log,
    e la sessione fallisce muta.  Costato una corsa a vuoto.
    """
    mc = "%s/%s.mc" % (DIR, nome)
    log = "%s/%s.log" % (DIR, nome)
    with open(mc, "w", encoding="ascii") as f:
        f.write("\n".join(comandi) + "\n")
    for p in (log, png):
        if os.path.exists(p):
            os.remove(p)

    r = subprocess.run(
        [vice(), "-pal", "+sound", "-warp",
         "-limitcycles", str(int(tetto)),
         "-monlog", "-monlogname", log,
         "-moncommands", mc,
         "-exitscreenshot", png,
         "-cartcrt", CRT],
        capture_output=True, text=True, timeout=900)

    if not os.path.exists(log):
        print("  [%s] VICE uscito con %d senza log" % (nome, r.returncode))
        print("  stdout:", (r.stdout or "")[-400:])
        print("  stderr:", (r.stderr or "")[-400:])
        return None
    testo = open(log, "r", encoding="latin1", errors="replace").read()
    trovati = re.findall(r"Stopwatch:\s+(\d+)", testo)
    return int(trovati[-1]) if trovati else None


def main():
    os.makedirs(DIR, exist_ok=True)
    if not os.path.exists(CRT):
        sys.exit("manca %s: generarlo con  python gen_testo.py --film  "
                 "e poi  .\\c64.ps1 genfilm -SoloCompila" % CRT)

    stati = []      # (parole_visibili, png, cicli_di_inizio)
    confini = []    # cicli a cui compare la parola k (k = 1..NGEN)
    orologio = 0

    for k in range(1, NGEN + 1):
        nome = "k%02d" % k
        png = "%s/stato%02d.png" % (DIR, k - 1)
        snap = "%s/s%02d.vsf" % (DIR, k)
        cmd = []
        if k > 1:
            cmd.append('undump "%s/s%02d.vsf"' % (DIR, k - 1))
        cmd += ["until " + STAMPA_TOKEN, "stopwatch", 'dump "%s"' % snap,
                "quit"]

        cron = sessione(nome, cmd, orologio + MARGINE, png)
        if cron is None or not os.path.exists(snap) or not os.path.exists(png):
            sys.exit("sessione %d fallita: guarda %s/%s.log" % (k, DIR, nome))

        # lo scatto e' lo stato con k-1 parole, valido DA quando e' comparsa
        # la parola k-1 (cioe' il confine precedente) FINO a questo confine
        stati.append({"parole": k - 1, "png": png})
        confini.append(cron)
        orologio = cron
        print("  parola %2d a %11d cicli  (%7.2f s)  [+%.2f s]"
              % (k, cron, cron / CLOCK,
                 (cron - confini[-2]) / CLOCK if len(confini) > 1
                 else cron / CLOCK), flush=True)

    # ultimo scatto: si riparte dall'ultimo snapshot e si lascia correre
    # quel tanto che basta a stampare la parola 34 e il "done."
    png = "%s/stato%02d.png" % (DIR, NGEN)
    cron = sessione("fine", ['undump "%s/s%02d.vsf"' % (DIR, NGEN)],
                    orologio + CODA, png)
    if not os.path.exists(png):
        sys.exit("scatto finale fallito")
    stati.append({"parole": NGEN, "png": png})

    fine = orologio + CODA
    with open("%s/film.json" % DIR, "w", encoding="utf-8") as f:
        json.dump({"clock": CLOCK, "confini": confini, "fine": fine,
                   "stati": stati}, f, indent=1)

    print("\n%d stati, ultimo confine %.2f s, totale %.2f s"
          % (len(stati), confini[-1] / CLOCK, fine / CLOCK))
    print("film/film.json scritto")


if __name__ == "__main__":
    main()
