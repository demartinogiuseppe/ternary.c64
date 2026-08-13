import sys

path = "train.py"
src = open(path, encoding="utf-8").read()

if "bitpatch" in src:
    print(">>> GIA' APPLICATO, non faccio nulla")
else:
    old = "    model = Transformer(gptconf)"
    new = "    model = Transformer(gptconf)\n    import bitpatch; bitpatch.apply(model)"
    if old not in src:
        print(">>> ERRORE: riga non trovata. FERMATI.")
        sys.exit(1)
    n = src.count(old)
    open(path, "w", encoding="utf-8").write(src.replace(old, new))
    print(f">>> OK: patch inserito in {n} punti")

# compatibilita' con PyTorch recenti
for f in ["train.py", "sample.py"]:
    s = open(f, encoding="utf-8").read()
    s2 = s.replace("torch.load(ckpt_path, map_location=device)",
                   "torch.load(ckpt_path, map_location=device, weights_only=False)")
    if s2 != s:
        open(f, "w", encoding="utf-8").write(s2)
        print(f">>> OK: torch.load sistemato in {f}")