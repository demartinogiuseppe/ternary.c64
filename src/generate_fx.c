/* generate_fx.c — inferenza interamente in virgola fissa, con TRONCAMENTO
   negli shift: la convenzione che il C64 implementa davvero.
   ATTENZIONE: questo file era una copia di generate_fx2.c (arrotondata), che
   il documento dichiarava scartata. Corretto.  Entrambe le convenzioni danno
   la sequenza identica al float, ma solo il troncamento combacia col C64.
   int16 Q5.10 ovunque, tabelle di lookup per SiLU, exp, 1/sqrt, cos/sin.
   Quantizzazione delle attivazioni tramite shift (niente divisioni). */

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
static fx    kcache[LAYERS][SEQ][KVDIM], vcache[LAYERS][SEQ][KVDIM];

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

int main(void) {
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
    const int N = (int)(sizeof(atteso) / sizeof(atteso[0])), NP = 5;

    static fx x[DIM], xb[DIM], q[DIM], kk[KVDIM], vv[KVDIM];
    static fx cat[DIM], out[DIM], h1[HIDDEN], h3[HIDDEN];
    static int32_t att[SEQ], logit[VOCAB];
    int ids[SEQ], n = NP, prima = -1, ndiv = 0;
    for (int i = 0; i < NP; i++) ids[i] = atteso[i];

    for (int step = 0; step < N; step++) {
        int pos   = (step < NP) ? step : n - 1;
        int token = ids[pos];

        for (int i = 0; i < DIM; i++)
            x[i] = clampfx(RSH((int32_t)emb[token][i] * emb_mant, emb_sh - QB));

        for (int l = 0; l < LAYERS; l++) {
            rmsnorm(xb, x, attn_norm[l], DIM);
            tmatvec(&mats[l][0], xb, q);
            tmatvec(&mats[l][1], xb, kk);
            tmatvec(&mats[l][2], xb, vv);

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
            memcpy(kcache[l][pos], kk, sizeof(kk));
            memcpy(vcache[l][pos], vv, sizeof(vv));

            for (int h = 0; h < HEADS; h++) {
                const fx *qh = q + h * HDIM;
                int32_t best = -0x7fffffff;
                for (int t = 0; t <= pos; t++) {
                    int32_t s = 0;
                    for (int d = 0; d < HDIM; d++) s += (int32_t)qh[d] * kcache[l][t][d];
                    att[t] = RSH(RSH(s, 2), QB);      /* /sqrt(16)=4, poi a Q5.10 */
                    if (att[t] > best) best = att[t];
                }
                int32_t somma = 0;
                for (int t = 0; t <= pos; t++) {
                    int32_t d = RSH(best - att[t], 4);
                    if (d > 511) { d = 511; sat_exp++; }
                    att[t] = T_exp[d];
                    somma += att[t];
                }
                if (somma < 1) somma = 1;
                fx *oh = cat + h * HDIM;
                for (int d = 0; d < HDIM; d++) {
                    int64_t acc = 0;
                    for (int t = 0; t <= pos; t++)
                        acc += (int64_t)att[t] * vcache[l][t][d];
                    oh[d] = clampfx((int32_t)(acc / somma));
                }
            }

            tmatvec(&mats[l][3], cat, out);
            for (int i = 0; i < DIM; i++) x[i] = clampfx((int32_t)x[i] + out[i]);

            rmsnorm(xb, x, ffn_norm[l], DIM);
            tmatvec(&mats[l][4], xb, h1);
            tmatvec(&mats[l][6], xb, h3);
            for (int i = 0; i < HIDDEN; i++) {
                int32_t k = RSH((int32_t)h1[i] * 24, QB) + 128;
                if (k < 0)   { k = 0;   sat_silu++; }
                if (k > 255) { k = 255; sat_silu++; }
                h1[i] = clampfx(RSH((int32_t)T_silu[k] * h3[i], QB));
            }
            tmatvec(&mats[l][5], h1, out);
            for (int i = 0; i < DIM; i++) x[i] = clampfx((int32_t)x[i] + out[i]);
        }

        rmsnorm(xb, x, final_norm, DIM);
        for (int t = 0; t < VOCAB; t++) {
            int32_t acc = 0;
            for (int d = 0; d < DIM; d++) acc += (int32_t)emb[t][d] * xb[d];
            logit[t] = acc;
        }

        if (step >= NP - 1 && n < N) {
            int best = 0;
            for (int t = 1; t < VOCAB; t++) if (logit[t] > logit[best]) best = t;
            if (best != atteso[n]) { if (prima < 0) prima = n; ndiv++; }
            ids[n++] = best;
        }
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