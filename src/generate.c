/* generate.c — inferenza completa: RoPE, KV cache, attenzione, generazione greedy.
   Attivazioni ancora in float: questo verifica la CATENA, non la quantizzazione. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define DIM     128
#define LAYERS    4
#define HEADS     8
#define HDIM     16
#define HALF      8          /* HDIM/2, coppie RoPE */
#define KVDIM    16
#define HIDDEN  352
#define VOCAB   512
#define SEQ     128

typedef struct { float scale; unsigned char *w; int rows, cols; } TMat;

static TMat  mats[LAYERS][7];
static float attn_norm[LAYERS][DIM], ffn_norm[LAYERS][DIM], final_norm[DIM];
static float emb_scale;
static int8_t emb[VOCAB][DIM];

static float kcache[LAYERS][SEQ][KVDIM], vcache[LAYERS][SEQ][KVDIM];
static float rcos[SEQ][HALF], rsin[SEQ][HALF];
static char  vocab[VOCAB][32];

#define WGT(m, i) ((int)(((m)->w[(i) >> 2] >> (((i) & 3) * 2)) & 3) - 1)

static void read_mat(FILE *f, TMat *m, int rows, int cols) {
    m->rows = rows; m->cols = cols;
    fread(&m->scale, 4, 1, f);
    m->w = malloc(rows * cols / 4);
    fread(m->w, 1, rows * cols / 4, f);
}

static void tmatvec(const TMat *m, const float *x, float *y) {
    for (int i = 0; i < m->rows; i++) {
        float acc = 0.0f;
        const int base = i * m->cols;
        for (int j = 0; j < m->cols; j++) {
            int w = WGT(m, base + j);
            if (w > 0)      acc += x[j];
            else if (w < 0) acc -= x[j];
        }
        y[i] = acc * m->scale;
    }
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ruota le coppie adiacenti (2i, 2i+1) con cos/sin della posizione */
static void rope(float *v, int nheads, int pos) {
    for (int h = 0; h < nheads; h++)
        for (int i = 0; i < HALF; i++) {
            float *p = v + h * HDIM + 2 * i;
            float r = p[0], im = p[1];
            p[0] = r * rcos[pos][i] - im * rsin[pos][i];
            p[1] = r * rsin[pos][i] + im * rcos[pos][i];
        }
}

static float x[DIM], xb[DIM], q[DIM], kk[KVDIM], vv[KVDIM];
static float cat[DIM], out[DIM], h1[HIDDEN], h3[HIDDEN];
static float att[SEQ], logits[VOCAB];

static void forward(int token, int pos) {
    for (int i = 0; i < DIM; i++) x[i] = emb[token][i] * emb_scale;

    for (int l = 0; l < LAYERS; l++) {
        rmsnorm(xb, x, attn_norm[l], DIM);
        tmatvec(&mats[l][0], xb, q);
        tmatvec(&mats[l][1], xb, kk);
        tmatvec(&mats[l][2], xb, vv);

        rope(q, HEADS, pos);
        rope(kk, 1, pos);

        memcpy(kcache[l][pos], kk, sizeof(kk));
        memcpy(vcache[l][pos], vv, sizeof(vv));

        const float inv = 1.0f / sqrtf((float)HDIM);
        for (int h = 0; h < HEADS; h++) {
            const float *qh = q + h * HDIM;
            for (int t = 0; t <= pos; t++) {
                float s = 0.0f;
                for (int d = 0; d < HDIM; d++) s += qh[d] * kcache[l][t][d];
                att[t] = s * inv;
            }
            softmax(att, pos + 1);
            float *oh = cat + h * HDIM;
            for (int d = 0; d < HDIM; d++) oh[d] = 0.0f;
            for (int t = 0; t <= pos; t++)
                for (int d = 0; d < HDIM; d++)
                    oh[d] += att[t] * vcache[l][t][d];
        }

        tmatvec(&mats[l][3], cat, out);
        for (int i = 0; i < DIM; i++) x[i] += out[i];

        rmsnorm(xb, x, ffn_norm[l], DIM);
        tmatvec(&mats[l][4], xb, h1);
        tmatvec(&mats[l][6], xb, h3);
        for (int i = 0; i < HIDDEN; i++)
            h1[i] = (h1[i] / (1.0f + expf(-h1[i]))) * h3[i];
        tmatvec(&mats[l][5], h1, out);
        for (int i = 0; i < DIM; i++) x[i] += out[i];
    }

    rmsnorm(xb, x, final_norm, DIM);
    for (int t = 0; t < VOCAB; t++) {
        float acc = 0.0f;
        for (int d = 0; d < DIM; d++) acc += emb[t][d] * xb[d];
        logits[t] = acc * emb_scale;
    }
}

/* sequenza greedy attesa, prodotta da make_seq.py */
static const int atteso[] = {
    1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 315, 421, 395, 317,
    426, 338, 401, 396, 267, 337, 410, 408, 419, 292, 411, 322, 265, 262, 379,
    419, 415, 271, 411, 426
};
#define N_ATTESO ((int)(sizeof(atteso) / sizeof(atteso[0])))
#define N_PROMPT 5

/* -seq <token> : parte da un token qualunque, stampa solo gli id generati.
   Serve a confrontare il float con le varianti in virgola fissa su PROMPT
   DIVERSI, non solo su quello di riferimento. */
int main(int argc, char **argv) {
    int seme = -1;
    if (argc > 2 && strcmp(argv[1], "-seq") == 0) seme = atoi(argv[2]);
    FILE *f = fopen("model_ternary.bin", "rb");
    if (!f) { printf("model_ternary.bin non trovato\n"); return 1; }
    char magic[5] = {0}; int hdr[7];
    fread(magic, 1, 4, f);
    if (strcmp(magic, "TC64")) { printf("magic errato\n"); return 1; }
    fread(hdr, 4, 7, f);

    const int shp[7][2] = {{DIM,DIM}, {KVDIM,DIM}, {KVDIM,DIM}, {DIM,DIM},
                           {HIDDEN,DIM}, {DIM,HIDDEN}, {HIDDEN,DIM}};
    for (int l = 0; l < LAYERS; l++)
        for (int k = 0; k < 7; k++) read_mat(f, &mats[l][k], shp[k][0], shp[k][1]);
    for (int l = 0; l < LAYERS; l++) {
        fread(attn_norm[l], 4, DIM, f);
        fread(ffn_norm[l],  4, DIM, f);
    }
    fread(final_norm, 4, DIM, f);
    fread(&emb_scale, 4, 1, f);
    fread(emb, 1, VOCAB * DIM, f);
    long letti = ftell(f); fseek(f, 0, SEEK_END);
    if (letti != ftell(f)) { printf("file DISALLINEATO\n"); return 1; }
    fclose(f);

    f = fopen("rope.bin", "rb");
    if (!f) { printf("rope.bin non trovato\n"); return 1; }
    fread(rcos, 4, SEQ * HALF, f);
    fread(rsin, 4, SEQ * HALF, f);
    fclose(f);

    f = fopen("vocab.txt", "rb");
    if (!f) { printf("vocab.txt non trovato\n"); return 1; }
    for (int i = 0; i < VOCAB; i++) {
        if (!fgets(vocab[i], sizeof(vocab[i]), f)) { vocab[i][0] = 0; continue; }
        char *p = strpbrk(vocab[i], "\r\n"); if (p) *p = 0;
    }
    fclose(f);
    printf("modello, rope e vocab caricati\n\n");

    int ids[SEQ];
    /* prompt di prova = BOS + un token, come il riferimento che inizia con
       BOS: partire da un token isolato mette il modello in uno stato che non
       si presenta mai davvero. */
    const int NP = seme >= 0 ? 2 : N_PROMPT;
    if (seme >= 0) { ids[0] = 1; ids[1] = seme; }
    else for (int i = 0; i < N_PROMPT; i++) ids[i] = atteso[i];
    int n = NP, errori = 0;

    /* consuma il prompt */
    for (int p = 0; p < NP; p++) forward(ids[p], p);

    /* genera greedy */
    while (n < N_ATTESO) {
        int best = 0;
        for (int t = 1; t < VOCAB; t++) if (logits[t] > logits[best]) best = t;
        if (seme < 0 && best != atteso[n]) {
            printf("DIVERGENZA a posizione %d: C=%d, atteso=%d\n", n, best, atteso[n]);
            errori++;
        }
        ids[n] = best;
        forward(best, n);
        n++;
    }

    if (seme >= 0) {
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n");
        return 0;
    }

    printf("token generati: ");
    for (int i = 0; i < n; i++) printf("%d ", ids[i]);
    printf("\n\ntesto: ");
    for (int i = 1; i < n; i++) printf("%s", vocab[ids[i]]);
    printf("\n\n%s\n", errori ? "*** SEQUENZE DIVERSE ***"
                              : "*** SEQUENZA IDENTICA A PYTORCH ***");
    return errori ? 1 : 0;
}