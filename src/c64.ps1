# c64.ps1 - compile a source, check its listing, and run the .crt in VICE
#           HEADLESS, saving the final screen as a .png next to the source.
#
# Why this exists. In cartridge mode there is no printf and the KERNAL never
# ran its init: the only way to know what happened is to look at the screen.
# Doing that by hand (attach, look, detach, recompile) is slow and has a trap:
# VICE holds the .crt open and stops oscar64 from overwriting it WITHOUT
# REPORTING AN ERROR, so you end up re-running the old binary and drawing
# conclusions from it. Here VICE exits on its own after a cycle cap, the file
# is never held open, and the .crt is deleted before recompiling.
#
#   .\c64.ps1 gen                  compile gen.c, check the listing, run it
#   .\c64.ps1 gen -Cicli 80000000  give it more machine time
#   .\c64.ps1 gen -SoloCompila     stop after checking the listing
#
# 1,000,000 cycles = 1 second of C64 time.
#
# ⚠️ Inline comments in this project are in Italian. The header of each file
#    is in English; the rest was written as the work went along.

param(
    [Parameter(Mandatory = $true)][string]$Nome,
    [long]$Cicli = 20000000,
    [switch]$SoloCompila
)

$ErrorActionPreference = "Stop"

# ---- dove stanno gli strumenti ------------------------------------------
# Ordine di ricerca: variabile d'ambiente, poi il percorso di default della
# macchina su cui il progetto e' nato, poi il PATH.  Se usi percorsi diversi
# non toccare questo file: basta
#     $env:OSCAR64 = "C:\...\oscar64.exe"
#     $env:VICE    = "C:\...\x64sc.exe"

function Trova-Strumento($daEnv, $default, $nomeExe) {
    if ($daEnv -and (Test-Path $daEnv)) { return $daEnv }
    if (Test-Path $default) { return $default }
    $c = Get-Command $nomeExe -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    return $null
}

$sorgenti = $PSScriptRoot
$oscar  = Trova-Strumento $env:OSCAR64 "C:\oscar64\bin\oscar64.exe" "oscar64.exe"
$vice   = Trova-Strumento $env:VICE "" "x64sc.exe"
$python = Trova-Strumento $env:PYTHON `
          (Join-Path (Split-Path $sorgenti -Parent) "venv\Scripts\python.exe") `
          "python.exe"

if (-not $oscar) {
    Write-Output "oscar64 non trovato. Imposta `$env:OSCAR64 al percorso di oscar64.exe"
    exit 1
}
if (-not $python) {
    Write-Output "python non trovato. Imposta `$env:PYTHON o mettilo nel PATH"
    exit 1
}
if (-not $vice -and -not $SoloCompila) {
    Write-Output "VICE non trovato. Imposta `$env:VICE al percorso di x64sc.exe,"
    Write-Output "oppure usa -SoloCompila per fermarti dopo la verifica del listato"
    exit 1
}

$c   = Join-Path $sorgenti "$Nome.c"
$crt = Join-Path $sorgenti "$Nome.crt"
$asm = Join-Path $sorgenti "$Nome.asm"
$png = Join-Path $sorgenti "$Nome.png"

if (-not (Test-Path $c)) { Write-Output "manca $c"; exit 1 }

# 1. via il .crt vecchio, cosi' un fallimento silenzioso non passa inosservato
if (Test-Path $crt) { Remove-Item $crt -Force }

# 2. compila
& $oscar -O2 -tf=crt "-o=$Nome.crt" "$Nome.c" 2>&1 | Write-Output
if ($LASTEXITCODE -ne 0) { Write-Output "COMPILAZIONE FALLITA"; exit 1 }
if (-not (Test-Path $crt)) { Write-Output "il .crt non e' stato prodotto"; exit 1 }
$info = Get-Item $crt
Write-Output ("compilato: {0} byte, {1}" -f $info.Length, $info.LastWriteTime)

# 3. verifica il listato: diramazioni a spostamento zero = binario da buttare
& $python (Join-Path $sorgenti "verifica_asm.py") $asm
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($SoloCompila) { exit 0 }

# 4. esegue senza interfaccia e fotografa lo schermo all'uscita
if (Test-Path $png) { Remove-Item $png -Force }
$log = Join-Path $env:TEMP "vice_$Nome.log"
Start-Process -FilePath $vice -Wait -NoNewWindow `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
    -ArgumentList @("-warp", "-limitcycles", "$Cicli",
                    "-exitscreenshot", $png, "-cartcrt", $crt)

if (Test-Path $png) {
    Write-Output "schermo salvato in $png"
} else {
    Write-Output "NESSUNO SCREENSHOT - guarda $log"
    exit 1
}
