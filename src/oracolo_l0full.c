/* oracolo_l0full.c — riferimento del LAYER 0 COMPLETO in virgola fissa
   (troncamento), con le somme di controllo di ogni stadio.

   Stadi verificabili, primo token (BOS=1, posizione 0):
       x0    embedding
       xb    RMSnorm(attention_norm)
       q,k,v tre matvec
       att   uscita dell'attenzione (a pos 0 = v ripetuto sulle 8 teste)
       o     wo(att)
       x1    x0 + o                       <- residua
       xb2   RMSnorm(ffn_norm)
       h1,h3 w1(xb2), w3(xb2)
       sw    SiLU(h1) * h3
       o2    w2(sw)
       x2    x1 + o2                      <- uscita del layer

   Nota: a posizione 0 RoPE e' l'identita' e il softmax e' banale. Quei due
   pezzi passano ma non vengono messi alla prova: serviranno piu' token.

   Compilare:  gcc -O2 -o oracolo_l0full.exe oracolo_l0full.c -lm */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define DIM 128
#define LAYERS 4
#define HEADS 8
#define HDIM 16
#define HIDDEN 352
#define KVDIM 16
#define VOCAB 512
#define QB 10

typedef int16_t fx;
typedef struct { int32_t mant; int sh; unsigned char *w; int rows, cols; } TMat;

static TMat  mats[LAYERS][7];
static fx    attn_norm[LAYERS][DIM], ffn_norm[LAYERS][DIM], final_norm[DIM];
static int32_t emb_mant; static int emb_sh;
static int8_t emb[VOCAB][DIM];
static int16_t T_silu[256], T_exp[512], T_rsq[256];

#define WGT(m, i) ((int)(((m)->w[(i) >> 2] >> (((i) & 3) * 2)) & 3) - 1)

static long sat = 0;

static fx clampfx(int32_t v)
{
    if (v >  32767) { sat++; return  32767; }
    if (v < -32768) { sat++; return -32768; }
    return (fx)v;
}

static void mkscale(float s, int32_t *mant, int *sh)
{
    int e = 0;
    while (s < 0.5f && e < 40) { s *= 2.0f; e++; }
    *mant = (int32_t)lrintf(s * 32768.0f);
    *sh   = 15 + e;
}

static void read_mat(FILE *f, TMat *m, int rows, int cols)
{
    float s;
    m->rows = rows; m->cols = cols;
    fread(&s, 4, 1, f);
    mkscale(s, &m->mant, &m->sh);
    m->w = malloc(rows * cols / 4);
    fread(m->w, 1, rows * cols / 4, f);
}

static int quant_shift(const fx *x, int n, int8_t *xq)
{
    int32_t mx = 0;
    for (int i = 0; i < n; i++) { int32_t a = abs(x[i]); if (a > mx) mx = a; }
    int s = 0;
    while ((mx >> s) > 127) s++;
    for (int i = 0; i < n; i++) xq[i] = (int8_t)(x[i] >> s);
    return s;
}

static void tmatvec(const TMat *m, const fx *x, fx *y)
{
    static int8_t xq[HIDDEN];
    int s = quant_shift(x, m->cols, xq);
    const int sh = m->sh - s;
    for (int i = 0; i < m->rows; i++) {
        int16_t a16 = 0;
        const int base = i * m->cols;
        for (int j = 0; j < m->cols; j++) {
            int w = WGT(m, base + j);
            if (w > 0)      a16 += xq[j];
            else if (w < 0) a16 -= xq[j];
        }
        int32_t v = (int32_t)a16 * m->mant;
        y[i] = clampfx(sh >= 0 ? (v >> sh) : (v << -sh));
    }
}

static void rmsnorm(fx *o, const fx *x, const fx *w, int n)
{
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
        int32_t t = ((int32_t)x[i] * inv) >> 14;
        o[i] = clampfx((t * w[i]) >> 12);
    }
}

static long somma(const fx *v, int n)
{
    long s = 0;
    for (int i = 0; i < n; i++) s += v[i];
    return s;
}

static void mostra(const char *nome, const fx *v, int n)
{
    printf("%-5s somma %8ld   primi 6:", nome, somma(v, n));
    for (int i = 0; i < 6 && i < n; i++) printf(" %d", v[i]);
    printf("\n");
}

int main(void)
{
    FILE *f = fopen("model_ternary.bin", "rb");
    if (!f) { printf("model_ternary.bin non trovato\n"); return 1; }
    char magic[5] = {0}; int hdr[7];
    fread(magic, 1, 4, f); fread(hdr, 4, 7, f);
    const int shp[7][2] = {{DIM,DIM},{KVDIM,DIM},{KVDIM,DIM},{DIM,DIM},
                           {HIDDEN,DIM},{DIM,HIDDEN},{HIDDEN,DIM}};
    for (int l = 0; l < LAYERS; l++)
        for (int k = 0; k < 7; k++) read_mat(f, &mats[l][k], shp[k][0], shp[k][1]);

    float tmp[DIM];
    for (int l = 0; l < LAYERS; l++) {
        fread(tmp, 4, DIM, f);
        for (int i = 0; i < DIM; i++) attn_norm[l][i] = (fx)lrintf(tmp[i] * 4096.0f);
        fread(tmp, 4, DIM, f);
        for (int i = 0; i < DIM; i++) ffn_norm[l][i]  = (fx)lrintf(tmp[i] * 4096.0f);
    }
    fread(tmp, 4, DIM, f);
    for (int i = 0; i < DIM; i++) final_norm[i] = (fx)lrintf(tmp[i] * 4096.0f);
    float es; fread(&es, 4, 1, f); mkscale(es, &emb_mant, &emb_sh);
    fread(emb, 1, VOCAB * DIM, f);
    fclose(f);

    f = fopen("tables.bin", "rb");
    if (!f) { printf("tables.bin non trovato\n"); return 1; }
    fread(T_silu, 2, 256, f);
    fread(T_exp,  2, 512, f);
    fread(T_rsq,  2, 256, f);
    fclose(f);

    fx x[DIM], xb[DIM], q[DIM], kk[KVDIM], vv[KVDIM];
    fx att[DIM], o[DIM], xb2[DIM], h1[HIDDEN], h3[HIDDEN], o2[DIM];

    printf("=== LAYER 0, token BOS=1, posizione 0 ===\n\n");

    for (int i = 0; i < DIM; i++)
        x[i] = clampfx(((int32_t)emb[1][i] * emb_mant) >> (emb_sh - QB));
    mostra("x0", x, DIM);

    rmsnorm(xb, x, attn_norm[0], DIM);
    mostra("xb", xb, DIM);

    tmatvec(&mats[0][0], xb, q);   mostra("q", q, DIM);
    tmatvec(&mats[0][1], xb, kk);  mostra("k", kk, KVDIM);
    tmatvec(&mats[0][2], xb, vv);  mostra("v", vv, KVDIM);

    /* posizione 0: RoPE identita', softmax di un elemento = 1.
       MQA: tutte le teste leggono lo stesso v. */
    for (int h = 0; h < HEADS; h++)
        for (int d = 0; d < HDIM; d++)
            att[h * HDIM + d] = vv[d];
    mostra("att", att, DIM);

    tmatvec(&mats[0][3], att, o);  mostra("o", o, DIM);

    for (int i = 0; i < DIM; i++) x[i] = clampfx((int32_t)x[i] + o[i]);
    mostra("x1", x, DIM);

    rmsnorm(xb2, x, ffn_norm[0], DIM);
    mostra("xb2", xb2, DIM);

    tmatvec(&mats[0][4], xb2, h1);  mostra("h1", h1, HIDDEN);
    tmatvec(&mats[0][6], xb2, h3);  mostra("h3", h3, HIDDEN);

    for (int i = 0; i < HIDDEN; i++) {
        int32_t k = (((int32_t)h1[i] * 24) >> QB) + 128;
        if (k < 0)   k = 0;
        if (k > 255) k = 255;
        h1[i] = clampfx(((int32_t)T_silu[k] * h3[i]) >> QB);
    }
    mostra("sw", h1, HIDDEN);

    tmatvec(&mats[0][5], h1, o2);  mostra("o2", o2, DIM);

    for (int i = 0; i < DIM; i++) x[i] = clampfx((int32_t)x[i] + o2[i]);
    mostra("x2", x, DIM);

    printf("\nsaturazioni Q5.10: %ld\n", sat);
    printf("\n--- somme da riprodurre sul C64 ---\n");
    printf("xb  %ld\nq   %ld\nk   %ld\nv   %ld\n",
           somma(xb, DIM), somma(q, DIM), somma(kk, KVDIM), somma(vv, KVDIM));
    printf("o   %ld\nxb2 %ld\nsw  %ld\no2  %ld\nx2  %ld\n",
           somma(o, DIM), somma(xb2, DIM), somma(h1, HIDDEN),
           somma(o2, DIM), somma(x, DIM));

    f = fopen("l0full.bin", "wb");
    fwrite(x, 2, DIM, f);          /* x2, l'uscita del layer */
    fclose(f);
    printf("\nl0full.bin scritto (x2, 256 byte)\n");
    return 0;
}