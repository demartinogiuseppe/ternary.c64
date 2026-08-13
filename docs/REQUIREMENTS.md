# Requirements

Everything below is **what this was actually built and verified with**, not a
minimum. Lower versions may well work; none of them were tried, and this
project has learned the hard way that a number obtained by inference is not a
measurement. So the versions here are provenance, not a floor.

**What replaces a minimum version is a test you can run.** If your toolchain
produces this, it is the right toolchain:

```
cd src
.\c64.ps1 l04 -Cicli 800000000     ->  336 comparisons, 0 errors
```

`l04` compares 21 stages at 16 consecutive positions against targets read from
the PC oracle. It covers the whole computation. If it passes, nothing about
your setup differs in a way that matters.

Built and verified on **Windows 10 Pro (10.0.19045)** with **Windows PowerShell
5.1**.

---

## 1. Running the cartridge

The smallest path, and the one most people want. The cartridge is
self-contained — model, tables, vocabulary and code are all inside it.

| | verified with |
|---|---|
| an emulator with **EasyFlash** support | **VICE 3.10**, `x64sc` |
| `gen.crt` | from [Releases](https://github.com/demartinogiuseppe/ternary.c64/releases) |

```
x64sc -cartcrt gen.crt
```

Nothing else. No Python, no compiler, no model file.

⚠️ The VICE binary carries no version metadata; **3.10** is the distribution it
came from (`GTK3VICE-3.10-win64`). It has to be `x64sc`, the cycle-exact C64
emulation — not `x64`.

---

## 2. Rebuilding from source

| | verified with | what needs it |
|---|---|---|
| **oscar64** | **1.32.272** | compiling every `.c` into a `.crt` |
| **Python** | **3.13.1** | the generators that write the `.c` files |
| **numpy** | **2.5.2** | `make_tables.py`, and only that |
| **VICE** | **3.10** (`x64sc`) | `c64.ps1` runs the result headless |
| **PowerShell** | **5.1** | `c64.ps1` |

oscar64 is at <https://github.com/drmortalwombat/oscar64>. ⚠️ **It prints no
version of its own** — `-h` gives usage and nothing else. The Windows installer
registers one (Add/Remove Programs shows `1.32.272`), and the binary that was
used here can be identified directly:

```
oscar64.exe  SHA-256  718294DB4008E00FCD9F729AC5479F96A8768DE21393F89F9E2EFA9A6E5134A9
```

If you are trying to work out whether an older oscar64 will do, the features
the build leans on are `-tf=crt`, `#embed`, and `#pragma region` /
`#pragma section` — the last two are what place code and weights into
EasyFlash banks. Whether a version missing them fails cleanly or quietly has
not been tried, which is another way of saying: run `l04`.

**numpy is the only non-standard-library import in the build path.** Every
other generator uses `struct`, `re`, `os`, `sys` and `subprocess` and nothing
else.

```
pip install numpy
```

If your tools live somewhere `c64.ps1` does not look, point at them instead of
editing the script:

```
$env:OSCAR64 = "C:\...\oscar64.exe"
$env:VICE    = "C:\...\x64sc.exe"
```

⚠️ **`-O2`, never `-O3`.** `-O3` enables `-Oz`, which places globals in zero
page and collides with the inline assembly. `c64.ps1` already passes `-O2`.

### For the verification only

| | verified with | what needs it |
|---|---|---|
| **gcc** | **16.2.0** (w64devkit) | the PC reference implementations |

The oracles are ordinary C programs that produce the targets the C64 output is
compared against. **The cartridge does not need them** — they are how you
check it, not how you build it.

```
gcc -O2 -o oracolo_att.exe oracolo_att.c -lm
```

### For the animation only

The GIF in the README is rebuilt by `cattura_film.py` (35 real screenshots, one
per word, driven by VICE's monitor) and `monta_gif.py` (the timer bar and the
GIF itself).

| | verified with | what needs it |
|---|---|---|
| **Pillow** | **12.3.0** | `monta_gif.py` |
| the C64 **chargen ROM** | the one shipped with VICE | drawing the timer in the machine's own font |

```
pip install Pillow
```

`monta_gif.py` looks for the character ROM inside the VICE installation; point
`$env:CHARGEN` at `C64/chargen-901225-01.bin` if it is somewhere else. Neither
script is needed to build or run the cartridge.

### Portability

The three generator commands in the README are plain Python and should run
anywhere. **`c64.ps1` has only ever been run under Windows PowerShell 5.1 on
Windows 10.** oscar64 and VICE both exist for Linux and macOS, so there is no
known obstacle — but "no known obstacle" is not a test result, and nobody has
run it.

---

## 3. Retraining the model

**You almost certainly do not need this.** `src/model_ternary.bin` is in the
repository — 242 KB, the trained ternary weights — and `export_banchi.py`
reads it with `struct`. The whole rebuild in section 2 works without ever
installing a deep learning framework.

This section is only for changing the model itself.

| | verified with | what needs it |
|---|---|---|
| **PyTorch** | **2.13.0+cu126** | `patch_train.py`, `bitpatch.py`, `export_ternary.py` |
| a fresh clone of **karpathy/llama2.c** | — | the training and sampling scripts, which are not redistributed here |

⚠️ **`sample.py` without the patch loads the latent weights and produces pure
noise.** Apply `patch_sample.py` and set `TERNARY=1`.

The PyTorch build recorded above is a CUDA one (`+cu126`). What training
actually costs — hardware, wall time, epochs — is **not stated here, because it
was never measured.** Everything else in this repository is a number that was
timed; this one would be a guess, so it is missing on purpose.

---

## 4. Real hardware

| | |
|---|---|
| **Kung Fu Flash 2** cartridge | ~40-50 € |
| a **microSD** card | |
| a **Commodore 64** | unexpanded — no RAM expansion of any kind |

⚠️ **This has not been done yet.** Everything in this repository was verified
in VICE, which is cycle-accurate. Nothing in the design depends on exact
timing — no raster tricks, no cycle-counted loops — so it is expected to work,
but until it has actually run on a real machine the hardware claim is
unverified.

This section will be corrected once it has been tried, **including if it
fails**. Until then, treat the hardware requirements above as the plan rather
than as a result: the only thing that has been confirmed is that the `.crt` is
a standard EasyFlash image, which is the format Kung Fu Flash reads.
