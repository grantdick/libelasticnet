#include <stdio.h>

#include <string.h>
#include <math.h>

#include "alloc.h"

void standardise_data(double **x, double *y, int N, int p,
                      double *xm, double *xs,
                      double *ym, double *ys)
{
    int i, j;
    double delta;

    *ym = *ys = 0;
    for (j = 0; j < p; ++j) xm[j] = xs[j] = 0.0;

    for (i = 0; i < N; ++i) {
        delta = y[i] - *ym;
        (*ym) += delta / (i + 1);
        (*ys) += delta * (y[i] - *ym);

        for (j = 0; j < p; ++j) {
            delta = x[i][j] - xm[j];
            xm[j] += delta / (i + 1);
            xs[j] += delta * (x[i][j] - xm[j]);
        }
    }

    if (N < 2) {
        fprintf(stderr, "Warning: fewer than two instances, cannot compute standard deviation of values.\n");
        *ys = NAN;
        for (j = 0; j < p; ++j) xs[j] = NAN;
    } else {
        delta = 1.0 / (double)(N);
        *ys = sqrt(delta * *ys);
        for (j = 0; j < p; ++j) xs[j] = sqrt(delta * xs[j]);
    }

    for (i = 0; i < N; ++i) {
        for (j = 0; j < p; ++j) {
            x[i][j] = (x[i][j] - xm[j]) / xs[j];
            if (fpclassify(x[i][j]) != FP_NORMAL) x[i][j] = 0;
        }
        y[i] = (y[i] - *ym) / *ys;
    }
}

double **new_matrix(int n, int m)
{
    int i;
    double **x;

    x = ALLOC(n, sizeof(double *), false);
    x[0] = ALLOC(n * m, sizeof(double), false);
    for (i = 0; i < n; ++i) x[i] = x[0] + i * m;
    for (i = 0; i < n*m; ++i) x[0][i] = 0.0;

    return x;
}

double **copy_matrix(double **x, int n, int m)
{
    int i;
    double **y;

    y = ALLOC(n, sizeof(double *), false);
    y[0] = ALLOC(n * m, sizeof(double), false);
    for (i = 0; i < n; ++i) {
        y[i] = y[0] + i * m;
        memcpy(y[i], x[i], m * sizeof(double));
    }

    return y;
}

void release_matrix(double **x)
{
    if (x == NULL) return;

    free(x[0]);
    free(x);
}

double *new_vector(int n)
{
    int i;
    double *x;

    x = ALLOC(n, sizeof(double), false);
    for (i = 0; i < n; ++i) x[i] = 0.0;

    return x;
}

double  *copy_vector(double *x, int n)
{
    double *y;

    y = ALLOC(n, sizeof(double), false);
    memcpy(y, x, n * sizeof(double));

    return y;
}

void release_vector(double *x)
{
    free(x);
}
