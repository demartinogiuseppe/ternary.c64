"""budget_ram.py - quanta RAM serve al forward pass completo sul C64.

La stima del §14 (cache KV ~3 KB) era sbagliata di un ordine di grandezza.
Il riferimento tiene  kcache[LAYERS][SEQ][KVDIM]  e  vcache[LAYERS][SEQ][KVDIM],
entrambi int16: con SEQ=128 sono 32 KB, contro i ~29,7 KB di RAM totale.
Non e' una svista di dettaglio: decide max_seq_len.

Qui il conto e' esplicito e parametrico. Rifallo ogni volta che una misura
sostituisce una stima:  python budget_ram.py
"""

DIM, LAYERS, HEADS, HDIM, KVDIM, HIDDEN, VOCAB = 128, 4, 8, 16, 16, 352, 512
HALF = HDIM // 2

RAM_INIZIO, RAM_FINE = 0x0900, 0x8000
RAM_TOTALE = RAM_FINE - RAM_INIZIO          # 30464 byte
PILA = 1024                                 # stack + sstack, con margine

# misurato in l0full.map, non stimato
ATTIVAZIONI = (256 +   # x
               256 +   # xb
               704 +   # xq   (HIDDEN interi)
               256 +   # q
               256 +   # cat  (concatenazione teste)
               256 +   # o
               704 +   # h1
               704 +   # h3
                32 +   # kk
                32)    # vv

PL_PH = 176 * 16 * 2                        # tabella coppie, byte bassi + alti
TABELLE = (256 + 512 + 256) * 2             # silu, exp, 1/sqrt
RMSNORM = (2 * LAYERS + 1) * DIM * 2        # 9 vettori Q3.12
CODICE_RAM = 10240                          # STIMA: orchestrazione + output head


def budget(seq, byte_kv=2, costanti_in_cartuccia=True, logit_in_ram=False):
    kv = LAYERS * seq * KVDIM * 2 * byte_kv          # k e v
    rope = seq * HALF * 2 * 2                        # cos e sin, Q1.14
    punteggi = seq * 4                               # att[SEQ] int32
    logit = VOCAB * 4 if logit_in_ram else 0

    voci = [
        ("cache KV",            kv),
        ("tabella PL/PH",       PL_PH),
        ("attivazioni",         ATTIVAZIONI),
        ("punteggi attenzione", punteggi),
        ("logit",               logit),
        ("codice in RAM",       CODICE_RAM),
        ("pila",                PILA),
    ]
    if not costanti_in_cartuccia:
        voci += [("tabelle lookup", TABELLE),
                 ("RoPE cos/sin", rope),
                 ("pesi RMSNorm", RMSNORM)]
    totale = sum(v for _, v in voci)
    return voci, totale, rope


def stampa(titolo, **kw):
    voci, totale, rope = budget(**kw)
    print("\n%s" % titolo)
    for nome, n in voci:
        if n:
            print("    %-22s %6d B" % (nome, n))
    avanzo = RAM_TOTALE - totale
    esito = "STA" if avanzo >= 0 else "NON STA"
    print("    %-22s %6d B su %d   avanzo %+d B   %s"
          % ("TOTALE", totale, RAM_TOTALE, avanzo, esito))
    if kw.get("costanti_in_cartuccia", True):
        print("    (in cartuccia: tabelle %d + RoPE %d + RMSNorm %d = %d B)"
              % (TABELLE, rope, RMSNORM, TABELLE + rope + RMSNORM))


if __name__ == "__main__":
    print("RAM utile 0x0900-0x8000 = %d byte (%.1f KB)"
          % (RAM_TOTALE, RAM_TOTALE / 1024))
    print("codice in RAM: %d B e' una STIMA, il pezzo piu' incerto del conto"
          % CODICE_RAM)

    stampa("SEQ=128, KV int16, tutto in RAM  (il piano del §14)",
           seq=128, byte_kv=2, costanti_in_cartuccia=False)
    stampa("SEQ=128, KV int16, costanti in cartuccia",
           seq=128, byte_kv=2)
    stampa("SEQ=64,  KV int16, costanti in cartuccia",
           seq=64, byte_kv=2)
    stampa("SEQ=64,  KV int8,  costanti in cartuccia",
           seq=64, byte_kv=1)
    stampa("SEQ=32,  KV int16, costanti in cartuccia",
           seq=32, byte_kv=2)
