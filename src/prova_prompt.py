"""prova_prompt.py - quanto costa ogni scelta di realizzazione, su molti prompt.

Fino a ieri tutto era verificato su UNA frase sola. Con margini sottili (§6)
non e' una garanzia: una scelta puo' passare per fortuna su un prompt e
cambiare il testo su un altro.

Confronta col RIFERIMENTO IN VIRGOLA MOBILE (generate.c) le varianti in
virgola fissa, partendo da molti token diversi:

  rif    cache int16, una divisione per elemento   (come generate_fx)
  recip  cache int16, un reciproco per testa
  int8   cache int8 arrotondata, una divisione per elemento
  c64    cache int8 arrotondata, un reciproco per testa   <- il piano

Quello che conta non e' che due forme in virgola fissa coincidano fra loro,
ma quanto ciascuna si scosta dal float.

    python prova_prompt.py [quanti]
"""

import os
import subprocess
import sys

# gcc deve stare nel PATH.  Se sta altrove:  set GCC_BIN=...\bin
if os.environ.get("GCC_BIN"):
    os.environ["PATH"] = os.environ["GCC_BIN"] + ";" + os.environ["PATH"]

VARIANTI = [
    ("rif",   ["-DCACHE8=0", "-DRECIP=0"]),
    ("recip", ["-DCACHE8=0", "-DRECIP=1"]),
    ("int8",  ["-DCACHE8=1", "-DRECIP=0", "-DQROUND=1"]),
    ("c64",   ["-DCACHE8=1", "-DRECIP=1", "-DQROUND=1"]),
]


def compila(exe, sorgente, definizioni):
    r = subprocess.run(["gcc", "-O2"] + definizioni + ["-o", exe, sorgente, "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("compilazione fallita (%s):\n%s" % (exe, r.stderr[:500]))
        sys.exit(1)


def sequenza(exe, seme):
    r = subprocess.run(["./" + exe, "-seq", str(seme)],
                       capture_output=True, text=True)
    righe = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    return righe[-1].split() if righe else []


quanti = int(sys.argv[1]) if len(sys.argv) > 1 else 60

print("compilo il riferimento float e le %d varianti..." % len(VARIANTI))
compila("rif_float.exe", "generate.c", [])
for nome, d in VARIANTI:
    compila("oa_%s.exe" % nome, "oracolo_att.c", d)

semi = [2 + (i * 509) // max(quanti - 1, 1) for i in range(quanti)]

conta = {n: 0 for n, _ in VARIANTI}
primo = {n: [] for n, _ in VARIANTI}

for s in semi:
    rif = sequenza("rif_float.exe", s)
    if not rif:
        print("il riferimento float non produce nulla per il seme %d" % s)
        sys.exit(1)
    for nome, _ in VARIANTI:
        seq = sequenza("oa_%s.exe" % nome, s)
        if seq == rif:
            conta[nome] += 1
        else:
            k = next((i for i, (a, b) in enumerate(zip(rif, seq)) if a != b),
                     min(len(rif), len(seq)))
            primo[nome].append(k)

print()
print("prompt provati: %d   (prompt = BOS + un token, 33 generati)" % len(semi))
print()
print("%-7s %-12s %-10s %s" % ("forma", "identiche", "quota", "divergenza media"))
for nome, _ in VARIANTI:
    n = conta[nome]
    m = (sum(primo[nome]) / len(primo[nome])) if primo[nome] else float("nan")
    quota = "%.0f%%" % (100.0 * n / len(semi))
    media = "-" if not primo[nome] else "posizione %.1f" % m
    print("%-7s %-12s %-10s %s" % (nome, "%d/%d" % (n, len(semi)), quota, media))

print()
print("La riga 'rif' e' il costo della sola virgola fissa: e' il fondo contro")
print("cui vanno letti gli altri. Le scelte in piu' costano la differenza.")
