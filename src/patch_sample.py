src = open("sample.py", encoding="utf-8").read()
if "bitpatch" in src:
    print(">>> GIA' APPLICATO")
else:
    out, done = [], False
    for ln in src.split("\n"):
        out.append(ln)
        if not done and "load_state_dict" in ln:
            ind = ln[:len(ln) - len(ln.lstrip())]
            out.append(ind + "import bitpatch; bitpatch.apply(model)")
            done = True
    if not done:
        print(">>> ERRORE: riga non trovata")
    else:
        open("sample.py", "w", encoding="utf-8").write("\n".join(out))
        print(">>> OK: bitpatch inserito in sample.py")