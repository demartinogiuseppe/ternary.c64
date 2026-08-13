/* oracolo_att.c - THE ORACLE. This is the reference every check on the C64
   is made against, and it runs on the PC.

   It is the same inference the Commodore does - int16 Q5.10 throughout,
   truncating shifts, lookup tables, int8 KV cache, one reciprocal per head -
   written plainly in C so that it can be trusted and read. Nothing here is
   optimised.

       oracolo_att.exe          check the sequence against the reference
       oracolo_att.exe -dump    layer 0 targets      (feeds gen_l0att.py)
       oracolo_att.exe -dump4   all four layers      (feeds gen_l04.py)
       oracolo_att.exe -seq N   start from BOS + token N and print the ids

   The generators read the targets from this program rather than having them
   written by hand, so the oracle and the C64 code cannot drift apart.

   ⚠️ It does NOT rebuild itself: after touching this file run
      gcc -O2 -o oracolo_att.exe oracolo_att.c -lm
      or you will silently generate targets from the old version.

   Note: comments below are in Italian.

   --- appunti originali ---

   ORACOLO DELL'ATTENZIONE.  Deriva da generate_fx.c.
   Inferenza interamente in virgola fissa int16 Q5.10, shift con TRONCAMENTO
   (la convenzione che il C64 implementa davvero), tabelle di lookup per
   SiLU, exp, 1/sqrt, cos/sin.

   Rispetto al riferimento adotta la forma REALIZZABILE SUL 6502:
     - cache KV a int8 (k>>7, v>>5) con ARROTONDAMENTO in scrittura
     - punteggi calcolati direttamente sulla cache int8
     - normalizzazione softmax con UN reciproco per testa invece di una
       divisione per elemento: da 512 divisioni per token a 32
     - nessun int64 in nessun punto
   Con queste scelte la sequenza dei 35 token e' IDENTICA al riferimento float.

   Uso:
       oracolo_att.exe            verifica la sequenza
       oracolo_att.exe -dump      bersagli bit-exact del layer 0 (gen_l0att.py)
       oracolo_att.exe -dump4     bersagli di TUTTI E QUATTRO i layer, quattro
                                  stadi ciascuno: xb, q, cat, x2 (gen_l04.py)

   Interruttori per gli esperimenti: -DCACHE8=0 -DRECIP=0 -DQROUND=0
   -DKSH=n -DVSH=n.  Vedi la tabella nel documento di progetto, §14. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define DIM 128
#define LAYERS 4
#define HEADS 8
#define HDIM 16
#define HALF 8
#define KVDIM 16
#define HIDDEN 352
#define VOCAB 512
/* SEQ = max_seq_len effettivo sul C64.  Deve coincidere con SEQ di
   make_tables.py: l'oracolo legge cos/sin da tables.bin ed e' bit-exact col
   C64 solo se le tabelle sono le stesse.  64 posizioni = 2048 byte di RoPE,
   che tengono tables.bin nei 4096 byte del piano di memoria. */
#define SEQ 64

#define QB 10                     /* bit frazionari: Q5.10 */

typedef int16_t fx;

typedef struct { int32_t mant; int sh; unsigned char *w; int rows, cols; } TMat;

static TMat  mats[LAYERS][7];
static fx    attn_norm[LAYERS][DIM], ffn_norm[LAYERS][DIM], final_norm[DIM];  /* Q3.12 */
static int32_t emb_mant; static int emb_sh;
static int8_t emb[VOCAB][DIM];
static fx    kcache[LAYERS][SEQ][KVDIM], vcache[LAYERS][SEQ][KVDIM];  /* non piu' usate */

/* --- due scelte di realizzazione, separabili per capire cosa costa ---
   CACHE8 : cache KV a int8 con shift fisso (§14)     0 = int16 pieno
   RECIP  : un solo reciproco per testa               0 = una divisione per
                                                          elemento, come il
                                                          riferimento        */
#ifndef CACHE8
#define CACHE8 1
#endif
#ifndef RECIP
#define RECIP 1
#endif

/* shift indipendenti: 0 = niente quantizzazione, resta int16 pieno.
   QROUND: arrotonda invece di troncare nella SOLA scrittura in cache
           (sul 6502 costa una addizione). */
#if CACHE8
#ifndef KSH
#define KSH 7
#endif
#ifndef VSH
#define VSH 5
#endif
#else
#undef KSH
#undef VSH
#define KSH 0
#define VSH 0
#endif

/* ⚠️ QROUND=1 e' OBBLIGATORIO.  Con troncamento (QROUND=0) la sequenza dei 35
   token diverge a posizione 21.  Con arrotondamento e' identica.  Verificato:
   k o v da soli reggono anche troncati, e' la coppia che sfora.  Sul 6502
   costa una addizione per elemento, 128 per token: irrilevante. */
#ifndef QROUND
#define QROUND 1
#endif

#if QROUND
#define QUANT(v, s) (((s) > 0) ? (((v) + (1 << ((s) - 1))) >> (s)) : (v))
#else
#define QUANT(v, s) (((s) > 0) ? ((v) >> (s)) : (v))
#endif

/* valore MEMORIZZATO, non riscalato: con CACHE8 sta in un int8 */
static int16_t kq[LAYERS][SEQ][KVDIM], vq[LAYERS][SEQ][KVDIM];
static int32_t peso[SEQ];

/* bersagli bit-exact: quante posizioni stampare (tutta la sequenza) */
#ifndef NDUMP
#define NDUMP 35
#endif
static int  dump = 0, dump4 = 0;
static long sc_somma = 0, sm_somma = 0;

static void stampa(const char *nome, const fx *v, int n)
{
    long s = 0;
    for (int i = 0; i < n; i++) s += v[i];
    printf("  %-4s %9ld   ", nome, s);
    for (int i = 0; i < 6 && i < n; i++) printf("%d ", v[i]);
    printf("\n");
}

static void stampa8_inutilizzata(const char *nome, const int8_t *v, int n)
{
    long s = 0;
    for (int i = 0; i < n; i++) s += v[i];
    printf("  %-4s %9ld   ", nome, s);
    for (int i = 0; i < 6 && i < n; i++) printf("%d ", v[i]);
    printf("\n");
}

/* stadi di -dump4: nome L<layer><stadio>, senza spazi, cosi' il generatore
   li legge con la stessa regex del -dump */
static void stampa4(int l, const char *st, const fx *v, int n)
{
    char nm[16];
    snprintf(nm, sizeof nm, "L%d%s", l, st);
    stampa(nm, v, n);
}

static int16_t T_silu[256];    /* Q5.10, passo 1/24 */
static int16_t T_exp[512];     /* Q0.15, exp(-d), passo 1/64 */
static int16_t T_rsq[256];     /* Q1.14, 1/sqrt(m), m in [1,4) */
static int16_t T_cos[SEQ][HALF], T_sin[SEQ][HALF];   /* Q1.14 */

static long sat_silu = 0, sat_exp = 0, sat_fx = 0, ovf_acc = 0;
static int  acc_max = 0;

#define WGT(m, i) ((int)(((m)->w[(i) >> 2] >> (((i) & 3) * 2)) & 3) - 1)

/* shift a destra con arrotondamento al piu' vicino */
#define RSH(v, s) (((s) > 0) ? ((v) >> (s)) : (v))   /* TRONCAMENTO */

static fx clampfx(int32_t v) {
    if (v >  32767) { sat_fx++; return  32767; }
    if (v < -32768) { sat_fx++; return -32768; }
    return (fx)v;
}

static void mkscale(float s, int32_t *mant, int *sh) {
    int e = 0;
    while (s < 0.5f && e < 40) { s *= 2.0f; e++; }
    *mant = (int32_t)lrintf(s * 32768.0f);
    *sh   = 15 + e;
}

static void read_mat(FILE *f, TMat *m, int rows, int cols) {
    float s;
    m->rows = rows; m->cols = cols;
    fread(&s, 4, 1, f);
    mkscale(s, &m->mant, &m->sh);
    m->w = malloc(rows * cols / 4);
    fread(m->w, 1, rows * cols / 4, f);
}

static int quant_shift(const fx *x, int n, int8_t *xq) {
    int32_t mx = 0;
    for (int i = 0; i < n; i++) { int32_t a = abs(x[i]); if (a > mx) mx = a; }
    int s = 0;
    while ((mx >> s) > 127) s++;
    for (int i = 0; i < n; i++) {
        int32_t v = RSH((int32_t)x[i], s);
        if (v >  127) v =  127;
        if (v < -128) v = -128;
        xq[i] = (int8_t)v;
    }
    return s;
}

static void tmatvec(const TMat *m, const fx *x, fx *y) {
    static int8_t xq[HIDDEN];
    int s = quant_shift(x, m->cols, xq);
    const int sh = m->sh - s;
    for (int i = 0; i < m->rows; i++) {
        int16_t a16 = 0; int32_t a32 = 0;
        const int base = i * m->cols;
        for (int j = 0; j < m->cols; j++) {
            int w = WGT(m, base + j);
            if (w > 0)      { a16 += xq[j]; a32 += xq[j]; }
            else if (w < 0) { a16 -= xq[j]; a32 -= xq[j]; }
        }
        if (a16 != a32) ovf_acc++;
        if (abs(a32) > acc_max) acc_max = abs(a32);
        int32_t v = (int32_t)a16 * m->mant;
        y[i] = clampfx(sh >= 0 ? RSH(v, sh) : (v << -sh));
    }
}

static void rmsnorm(fx *o, const fx *x, const fx *w, int n) {
    int32_t ss = 0;
    for (int i = 0; i < n; i++) ss += (int32_t)x[i] * x[i];
    ss /= n;
    if (ss < 1) ss = 1;

    int e = 0;
    while (ss <  (1 << 20)) { ss <<= 2; e++; }
    while (ss >= (1 << 22)) { ss >>= 2; e--; }
    int k = (int)(((ss - (1 << 20)) * 256) / (3 << 20));
    if (k > 255) k = 255;
    int32_t inv = (int32_t)T_rsq[k] << e;

    for (int i = 0; i < n; i++) {
        int32_t t = RSH((int32_t)x[i] * inv, 14);
        o[i] = clampfx(RSH(t * w[i], 12));
    }
}

int main(int argc, char **argv) {
    int seme = -1;
    if (argc > 1 && strcmp(argv[1], "-dump") == 0) dump = 1;
    if (argc > 1 && strcmp(argv[1], "-dump4") == 0) dump4 = 1;
    if (argc > 2 && strcmp(argv[1], "-seq") == 0) seme = atoi(argv[2]);
    FILE *f = fopen("model_ternary.bin", "rb");
    if (!f) { printf("model_ternary.bin non trovato\n"); return 1; }
    char magic[5] = {0}; int hdr[7];
    fread(magic, 1, 4, f); fread(hdr, 4, 7, f);
    const int shp[7][2] = {{DIM,DIM},{KVDIM,DIM},{KVDIM,DIM},{DIM,DIM},
                           {HIDDEN,DIM},{DIM,HIDDEN},{HIDDEN,DIM}};
    for (int l = 0; l < LAYERS; l++)
        for (int k = 0; k < 7; k++) read_mat(f, &mats[l][k], shp[k][0], shp[k][1]);

    float tmp[DIM], wmax = 0.0f;
    for (int l = 0; l < LAYERS; l++) {
        fread(tmp, 4, DIM, f);
        for (int i = 0; i < DIM; i++) {
            if (fabsf(tmp[i]) > wmax) wmax = fabsf(tmp[i]);
            attn_norm[l][i] = (fx)lrintf(tmp[i] * 4096.0f);
        }
        fread(tmp, 4, DIM, f);
        for (int i = 0; i < DIM; i++) {
            if (fabsf(tmp[i]) > wmax) wmax = fabsf(tmp[i]);
            ffn_norm[l][i] = (fx)lrintf(tmp[i] * 4096.0f);
        }
    }
    fread(tmp, 4, DIM, f);
    for (int i = 0; i < DIM; i++) {
        if (fabsf(tmp[i]) > wmax) wmax = fabsf(tmp[i]);
        final_norm[i] = (fx)lrintf(tmp[i] * 4096.0f);
    }
    float es; fread(&es, 4, 1, f); mkscale(es, &emb_mant, &emb_sh);
    fread(emb, 1, VOCAB * DIM, f);
    fclose(f);
    printf("peso RMSNorm max = %.4f  (Q3.12 regge fino a 8.0)\n", wmax);

    f = fopen("tables.bin", "rb");
    if (!f) { printf("tables.bin non trovato\n"); return 1; }
    fread(T_silu, 2, 256, f);
    fread(T_exp,  2, 512, f);
    fread(T_rsq,  2, 256, f);
    fread(T_cos,  2, SEQ * HALF, f);
    fread(T_sin,  2, SEQ * HALF, f);
    fclose(f);

    static char vocab[VOCAB][32];
    f = fopen("vocab.txt", "rb");
    if (!f) { printf("vocab.txt non trovato\n"); return 1; }
    for (int i = 0; i < VOCAB; i++) {
        if (!fgets(vocab[i], sizeof(vocab[i]), f)) { vocab[i][0] = 0; continue; }
        char *p = strpbrk(vocab[i], "\r\n"); if (p) *p = 0;
    }
    fclose(f);

    static const int atteso[] = {
        1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 315, 421, 395, 317,
        426, 338, 401, 396, 267, 337, 410, 408, 419, 292, 411, 322, 265, 262, 379,
        419, 415, 271, 411, 426
    };
    /* modalita' sequenza: parte da un token qualunque e stampa gli id
       generati, senza confrontarli con atteso[].  Serve a verificare che le
       scelte di realizzazione (cache int8, reciproco unico) diano lo stesso
       testo su PROMPT DIVERSI, non solo su quello di riferimento. */
    const int N = (int)(sizeof(atteso) / sizeof(atteso[0]));
    const int NP = seme >= 0 ? 2 : 5;   /* prova = BOS + un token */

    static fx x[DIM], xb[DIM], q[DIM], kk[KVDIM], vv[KVDIM];
    static fx cat[DIM], out[DIM], h1[HIDDEN], h3[HIDDEN];
    static int32_t att[SEQ], logit[VOCAB];
    int ids[SEQ], n = NP, prima = -1, ndiv = 0;
    if (seme >= 0) { ids[0] = 1; ids[1] = seme; }
    else for (int i = 0; i < NP; i++) ids[i] = atteso[i];

    for (int step = 0; step < N; step++) {
        int pos   = (step < NP) ? step : n - 1;
        int token = ids[pos];

        for (int i = 0; i < DIM; i++)
            x[i] = clampfx(RSH((int32_t)emb[token][i] * emb_mant, emb_sh - QB));

        if (dump && pos < NDUMP) {
            printf("\n--- layer 0, posizione %d, token %d ---\n", pos, token);
            stampa("x0", x, DIM);
        }
        if (dump4 && pos < NDUMP) {
            printf("\n--- pos %d, token %d ---\n", pos, token);
            stampa("x0", x, DIM);
        }

        for (int l = 0; l < LAYERS; l++) {
            rmsnorm(xb, x, attn_norm[l], DIM);
            tmatvec(&mats[l][0], xb, q);
            tmatvec(&mats[l][1], xb, kk);
            tmatvec(&mats[l][2], xb, vv);
            if (dump && l == 0 && pos < NDUMP) {
                stampa("xb", xb, DIM);
                stampa("q", q, DIM);
                stampa("k", kk, KVDIM);
                stampa("v", vv, KVDIM);
            }
            if (dump4 && pos < NDUMP) {
                stampa4(l, "xb", xb, DIM);
                stampa4(l, "q",  q,  DIM);     /* prima di RoPE */
            }

            for (int h = 0; h < HEADS; h++)
                for (int i = 0; i < HALF; i++) {
                    fx *p = q + h * HDIM + 2 * i;
                    int32_t r = p[0], im = p[1];
                    p[0] = clampfx(RSH(r * T_cos[pos][i] - im * T_sin[pos][i], 14));
                    p[1] = clampfx(RSH(r * T_sin[pos][i] + im * T_cos[pos][i], 14));
                }
            for (int i = 0; i < HALF; i++) {
                fx *p = kk + 2 * i;
                int32_t r = p[0], im = p[1];
                p[0] = clampfx(RSH(r * T_cos[pos][i] - im * T_sin[pos][i], 14));
                p[1] = clampfx(RSH(r * T_sin[pos][i] + im * T_cos[pos][i], 14));
            }
            if (dump && l == 0 && pos < NDUMP) {
                stampa("qr", q, DIM);         /* dopo RoPE */
                stampa("kr", kk, KVDIM);
                sc_somma = sm_somma = 0;
            }

            /* ---- scrittura in cache ---- */
            for (int d = 0; d < KVDIM; d++) {
                int32_t a = QUANT((int32_t)kk[d], KSH);
                int32_t b = QUANT((int32_t)vv[d], VSH);
                if (KSH > 0) { if (a >  127) a =  127; if (a < -128) a = -128; }
                if (VSH > 0) { if (b >  127) b =  127; if (b < -128) b = -128; }
                kq[l][pos][d] = (int16_t)a;
                vq[l][pos][d] = (int16_t)b;
            }
            if (dump && l == 0 && pos < NDUMP) {
                stampa("kc", kq[l][pos], KVDIM);
                stampa("vc", vq[l][pos], KVDIM);
            }

            /* ---- attenzione nella forma realizzabile sul 6502 ----
               Il riferimento usava int64 e UNA DIVISIONE PER OGNI ELEMENTO
               di uscita: 16 per testa, 128 per layer, 512 per token. Sul 6502
               e' insostenibile. Qui:
                 - i punteggi si calcolano direttamente sulla cache int8,
                   e lo shift della cache si fonde nello shift finale:
                   (s << KSH) >> 2 >> QB  ==  s >> (QB + 2 - KSH)  ==  s >> 5
                 - la normalizzazione usa UN SOLO reciproco per testa:
                   inv = 2^30 / somma, poi w[t] = (att[t] * inv) >> 15.
                   Da 512 divisioni a 32 per token.
                 - tutto resta a 32 bit: somma >= T_exp[0] = 32767 sempre
                   (il massimo da' distanza 0), quindi inv <= 32770 e
                   att * inv <= 32767 * 32770 < 2^31.                          */
            for (int h = 0; h < HEADS; h++) {
                const fx *qh = q + h * HDIM;
                int32_t best = -0x7fffffff;
                for (int t = 0; t <= pos; t++) {
                    int32_t s = 0;
                    for (int d = 0; d < HDIM; d++)
                        s += (int32_t)qh[d] * kq[l][t][d];
                    att[t] = s >> (QB + 2 - KSH);
                    if (att[t] > best) best = att[t];
                    if (dump && l == 0 && pos < NDUMP) sc_somma += att[t];
                }
                int32_t somma = 0;
                for (int t = 0; t <= pos; t++) {
                    int32_t d = (best - att[t]) >> 4;
                    if (d > 511) { d = 511; sat_exp++; }
                    att[t] = T_exp[d];
                    somma += att[t];
                }
                if (somma < 1) somma = 1;
                fx *oh = cat + h * HDIM;
#if RECIP
                int32_t inv = 0x40000000 / somma;
                for (int t = 0; t <= pos; t++) {
                    peso[t] = (att[t] * inv) >> 15;
                    if (dump && l == 0 && pos < NDUMP) sm_somma += peso[t];
                }
                for (int d = 0; d < HDIM; d++) {
                    int32_t acc = 0;
                    for (int t = 0; t <= pos; t++)
                        acc += peso[t] * vq[l][t][d];
                    oh[d] = clampfx(acc >> (15 - VSH));
                }
#else
                for (int t = 0; t <= pos; t++)
                    if (dump && l == 0 && pos < NDUMP) sm_somma += att[t];
                for (int d = 0; d < HDIM; d++) {
                    int64_t acc = 0;
                    for (int t = 0; t <= pos; t++)
                        acc += (int64_t)att[t] * ((int32_t)vq[l][t][d] << VSH);
                    oh[d] = clampfx((int32_t)(acc / somma));
                }
#endif
            }

            if (dump && l == 0 && pos < NDUMP) {
                printf("  %-4s %9ld   (punteggi grezzi, tutte le teste)\n",
                       "sc", sc_somma);
                printf("  %-4s %9ld   (pesi softmax, atteso ~%d)\n",
                       "sm", sm_somma, HEADS * 32768);
                stampa("cat", cat, DIM);
            }
            if (dump4 && pos < NDUMP) stampa4(l, "cat", cat, DIM);

            tmatvec(&mats[l][3], cat, out);
            for (int i = 0; i < DIM; i++) x[i] = clampfx((int32_t)x[i] + out[i]);
            if (dump && l == 0 && pos < NDUMP) stampa("o", out, DIM);

            rmsnorm(xb, x, ffn_norm[l], DIM);
            if (dump && l == 0 && pos < NDUMP) stampa("xb2", xb, DIM);
            tmatvec(&mats[l][4], xb, h1);
            tmatvec(&mats[l][6], xb, h3);
            for (int i = 0; i < HIDDEN; i++) {
                int32_t k = RSH((int32_t)h1[i] * 24, QB) + 128;
                if (k < 0)   { k = 0;   sat_silu++; }
                if (k > 255) { k = 255; sat_silu++; }
                h1[i] = clampfx(RSH((int32_t)T_silu[k] * h3[i], QB));
            }
            if (dump && l == 0 && pos < NDUMP) stampa("sw", h1, HIDDEN);
            tmatvec(&mats[l][5], h1, out);
            for (int i = 0; i < DIM; i++) x[i] = clampfx((int32_t)x[i] + out[i]);
            if (dump && l == 0 && pos < NDUMP) {
                stampa("o2", out, DIM);
                stampa("x2", x, DIM);
            }
            if (dump4 && pos < NDUMP) stampa4(l, "x2", x, DIM);
        }

        rmsnorm(xb, x, final_norm, DIM);
        for (int t = 0; t < VOCAB; t++) {
            int32_t acc = 0;
            for (int d = 0; d < DIM; d++) acc += (int32_t)emb[t][d] * xb[d];
            logit[t] = acc;
        }
        if (dump4 && pos < NDUMP) {
            /* bersagli dell'output head: xb finale, somma dei logit e
               argmax.  La somma prende tutti e 512 i logit, quindi un solo
               numero sbagliato la sposta; l'argmax e' il token generato. */
            long slg = 0;
            int am = 0;
            for (int t = 0; t < VOCAB; t++) {
                slg += logit[t];
                if (logit[t] > logit[am]) am = t;
            }
            stampa("xbf", xb, DIM);
            printf("  %-4s %9ld\n", "lg0", (long)logit[0]);
            printf("  %-4s %9ld\n", "lg", slg);
            printf("  %-4s %9d\n", "am", am);
        }

        if (step >= NP - 1 && n < N) {
            int best = 0;
            for (int t = 1; t < VOCAB; t++) if (logit[t] > logit[best]) best = t;
            if (seme < 0 && best != atteso[n]) {
                if (prima < 0) prima = n;
                ndiv++;
            }
            ids[n++] = best;
        }
    }

    if (seme >= 0) {
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n");
        return 0;
    }

    printf("\n=== VIRGOLA FISSA, SHIFT ARROTONDATI ===\n");
    printf("accumulatore max   : %d  (limite int16 32767)\n", acc_max);
    printf("overflow acc int16 : %ld %s\n", ovf_acc, ovf_acc ? "<<< PROBLEMA" : "");
    printf("saturazioni Q5.10  : %ld %s\n", sat_fx, sat_fx ? "<<< da guardare" : "");
    printf("saturazioni SiLU   : %ld\n", sat_silu);
    printf("saturazioni exp    : %ld  (normale)\n", sat_exp);

    printf("\ntesto:");
    for (int i = 1; i < n; i++) printf("%s", vocab[ids[i]]);
    printf("\nriferimento:");
    for (int i = 1; i < N; i++) printf("%s", vocab[atteso[i]]);
    printf("\n\n");
    if (prima < 0) printf("*** SEQUENZA IDENTICA ***\n");
    else printf("prima divergenza a posizione %d di %d (%d token diversi)\n",
                prima, N, ndiv);
    return 0;
}