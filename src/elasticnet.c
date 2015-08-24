#include <stdio.h>
#include <stdbool.h>

#include <math.h>
#include <string.h>

#include "elasticnet.h"

#include "alloc.h"
#include "util.h"

#define ALPHA_THRESHOLD 1e-3
#define LAMBDA_TOLERANCE 1e-4
#define LAMBDA_EPSILON 1e-4
#define CONV_THRESHOLD 1e-7

static struct elasticnet_info *create_empty_model(int p, int K)
{
    struct elasticnet_info *l;

    l = ALLOC(1, sizeof(struct elasticnet_info), false);

    l->p = p;
    l->K = K;

    l->lambda     = new_vector(K);
    l->a0         = new_vector(K);
    l->beta       = new_matrix(K, p);
    l->rsqr       = new_vector(K);
    l->n_zero     = ALLOC(K, sizeof(int), true);

    l->xm         = new_vector(p);
    l->xs         = new_vector(p);

    l->n_folds    = 0;
    l->cv_mse     = NULL;
    l->cv_se      = NULL;
    l->cv_mse_min = -1;
    l->cv_mse_1se = -1;

    return l;
}



static double *compute_initial_gradient(double **X, double *y, int N, int p)
{
    int i, j;
    double *g;

    g = new_vector(p);

    for (j = 0; j < p; ++j) {
        for (i = 0; i < N; ++i) g[j] += X[i][j] * y[i];
    }

    return g;
}


static double **compute_covariance_matrix(double **X, int N, int p)
{
    double **c;
    int i, j, k;

    c = new_matrix(p, p);
    for (j = 0; j < p; ++j) {
        c[j][j] = 1.0;
        for (k = j + 1; k < p; ++k) {
            for (i = 0; i < N; ++i) c[k][j] += X[i][k] * X[i][j];
            c[j][k] = c[k][j];
        }
    }

    return c;
}



static void compute_lambda_path(double *lambda, int K,
                                double alpha, double epsilon,
                                double *g, int p)
{
    int i, j;
    double alf;

    /* from Friedman et al. (2010), lambda_{max} = max((abs(xy))/(N*alpha),
     * which is equal to: lambda_{max} = max(abs(g))/alpha */
    lambda[0] = 0;
    for (j = 0; j < p; ++j) if (fabs(g[j]) > lambda[0]) lambda[0] = fabs(g[j]);
    lambda[0] /= (alpha < ALPHA_THRESHOLD) ? ALPHA_THRESHOLD : alpha;

    /* the remaining values along the path are computed on a log scale
     * from lambda_{max} to lambda_{min) (= epsilon * lambda_{max}) */
    alf = pow(epsilon, 1.0 / (double)(K - 1));
    for (i = 1; i < K; ++i) lambda[i] = lambda[i - 1] * alf;
}


static double soft_threshold_update(int j, double *beta, double *g, double lambda, double alpha)
{
    double d, u, v;

    d = 1 + lambda * (1 - alpha);
    u = g[j] + beta[j];
    v = (u < 0 ? -u : u) - lambda * alpha;
    return (v < 0) ? 0 : (u < 0) ? -v / d : v / d;
}



static double elasticnet_coord_descent(double *beta,
                                       double lambda, double alpha,
                                       int N, int p, double rsqr,
                                       double **c, double *g)
{
    int j, k;
    double v, max_delta, delta;

    /* this is a slight deviation from how glmnet and lasso4j work, in
     * that it does not perfom sparse updates (i.e., it does not
     * acknowledge coefficients that are zero. Essentially, it is the
     * "covariance updates" method proposed by Friedman et al. (2010),
     * but done less efficiently (for the sake of code simplicity) */
    do {
        max_delta = 0;
        for (j = 0; j < p; ++j) {
            if ((v = soft_threshold_update(j, beta, g, lambda, alpha)) == beta[j]) continue;

            /* coefficient has changed, so update model and stats */
            delta = v - beta[j];

            beta[j] = v;
            rsqr += delta * (2.0 * g[j] - delta);
            for (k = 0; k < p; ++k) g[k] -= c[j][k] * delta;

            delta = delta * delta;
            if (delta > max_delta) max_delta = delta;
        }
    } while (max_delta > CONV_THRESHOLD);

    return rsqr;
}









struct elasticnet_info *elasticnet_cv_fit(double **X, double *y, int N, int p,
                                          int *fold_id, int n_folds,
                                          double *lambda, int K, double alpha)
{
    int i, j, f;

    struct elasticnet_info *l, *l_fold;
    double **train_X, *train_y, **test_X, *test_y;
    int N_train, N_test;
    bool local_folds;

    double *yhat, r, mse, delta;

    /* first, fit a model over the entire set of data and init the
     * required cross-validation structures */
    l = elasticnet_fit(X, y, N, p, lambda, K, alpha);

    l->n_folds = n_folds;
    l->cv_mse = new_vector(l->K);
    l->cv_se  = new_vector(l->K);

    train_X = ALLOC(N, sizeof(double *), false);
    test_X  = ALLOC(N, sizeof(double *), false);
    train_y = new_vector(N);
    test_y  = new_vector(N);

    yhat = new_vector(N);

    local_folds = fold_id == NULL;
    if (local_folds) {
        fold_id = ALLOC(N, sizeof(int), false);
        for (i = 0; i < N; ++i) fold_id[i] = i % n_folds;
    }

    /* then, iterate over each fold */
    for (f = 0; f < n_folds; ++f) {
        N_train = N_test = 0;
        for (i = 0; i < N; ++i) {
            if (fold_id[i] == f) {
                test_X[N_test] = X[i];
                test_y[N_test] = y[i];
                N_test++;
            } else {
                train_X[N_train] = X[i];
                train_y[N_train] = y[i];
                N_train++;
            }
        }

        l_fold = elasticnet_fit(train_X, train_y, N_train, l->p, l->lambda, l->K, alpha);

        for (i = 0; i < l->K; ++i) {
            elasticnet_predict(l_fold, test_X, N_test, l->lambda[i], yhat);

            mse = 0;
            for (j = 0; j < N_test; ++j) {
                r = test_y[j] - yhat[j];
                mse += (r*r - mse) / (j + 1);
            }

            delta = mse - l->cv_mse[i];
            l->cv_mse[i] += delta / (f + 1);
            l->cv_se[i]  += delta * (mse - l->cv_mse[i]);
        }

        elasticnet_free(l_fold);
    }

    for (i = 0; i < l->K; ++i) l->cv_se[i] = sqrt(l->cv_se[i] / (n_folds * (n_folds - 1)));

    l->cv_mse_min = 0;
    for (i = 1; i < l->K; ++i) if (l->cv_mse[i] < l->cv_mse[l->cv_mse_min]) l->cv_mse_min = i;
    delta = l->cv_mse[l->cv_mse_min] + l->cv_se[l->cv_mse_min];
    l->cv_mse_1se = l->cv_mse_min;
    for (i = l->cv_mse_1se; i > 0; --i) if (l->cv_mse[i] < delta) l->cv_mse_1se = i;

    release_vector(yhat);

    free(train_X);
    free(test_X);
    release_vector(train_y);
    release_vector(test_y);

    if (local_folds) free(fold_id);

    return l;
}

struct elasticnet_info *elasticnet_fit(double **X, double *y, int N, int p,
                                       double *lambda, int K, double alpha)
{
    int i, j;
    struct elasticnet_info *l;
    double **sX, *sy;
    double **c, *g;
    double scale;
    double *prev_beta, prev_rsqr;

    l = create_empty_model(p, K);

    sX = copy_matrix(X, N, p);
    sy = copy_vector(y, N);

    standardise_data(sX, sy, N, p, l->xm, l->xs, &(l->ym), &(l->ys));

    /* a further scaling of the standardised data is performed (divide
     * through by sqrt(N)), as this makes later updating of the
     * supporting arrays easier (and also faster) */
    scale = 1.0 / sqrt(N);
    for (i = 0; i < N; ++i) {
        for (j = 0; j < p; ++j) sX[i][j] *= scale;
        sy[i] *= scale;
    }

    g = compute_initial_gradient(sX, sy, N, p);
    c = compute_covariance_matrix(sX, N, p);

    /* ensure that the regularisation path is correctly defined */
    if (lambda == NULL) {
        compute_lambda_path(l->lambda, K, alpha, LAMBDA_EPSILON, g, p);
    } else {
        /* copy and standardise the supplied lambda values */
        for (i = 0; i < l->K; ++i) l->lambda[i] = lambda[i] / l->ys;
    }

    prev_rsqr = 0.0;
    prev_beta = new_vector(p);
    for (j = 0; j < p; ++j) prev_beta[j] = 0.0;

    for (i = 0; i < K; ++i) {
        /* warm start - start from the previous set of coefficients */
        memcpy(l->beta[i], prev_beta, p * sizeof(double));

        /* fit the model using the current lambda penalty factor */
        l->rsqr[i] = elasticnet_coord_descent(l->beta[i], l->lambda[i], alpha, N, p, prev_rsqr, c, g);

        /* identify the number of terms that have entered the model */
        l->n_zero[i] = 0;
        for (j = 0; j < p; ++j) if (l->beta[i][j] == 0) l->n_zero[i]++;

        /* stop early if the regularisation path is not going to lead
         * to significant improvements in R^2 */
        if ((lambda == NULL) && ((l->rsqr[i] - prev_rsqr) < (CONV_THRESHOLD * l->rsqr[i]))) {
            l->K = i;
            break;
        }

        memcpy(prev_beta, l->beta[i], p * sizeof(double));
        prev_rsqr = l->rsqr[i];
    }

    release_vector(prev_beta);

    /* finally, update coefficients and parameters to their unscaled
     * equivalents */
    for (i = 0; i < l->K; ++i) {
        l->lambda[i] *= l->ys;

        for (j = 0; j < l->p; ++j) if (l->xs[j] != 0) l->beta[i][j] *= l->ys / l->xs[j];

        l->a0[i] = l->ym;
        for (j = 0; j < l->p; ++j) l->a0[i] -= l->beta[i][j] * l->xm[j];
    }

    release_matrix(sX);
    release_vector(sy);
    release_vector(g);
    release_matrix(c);

    return l;
}



void elasticnet_predict(struct elasticnet_info *model, double **X, int N, double lambda, double *yhat)
{
    int i, j;
    double *beta, a0;

    a0 = NAN;
    beta = new_vector(model->p);

    elasticnet_coefficients(model, lambda, beta, &a0);

    for (i = 0; i < N; ++i) {
        yhat[i] = a0;
        for (j = 0; j < model->p; ++j) yhat[i] += beta[j] * X[i][j];
    }

    release_vector(beta);
}



void elasticnet_free(struct elasticnet_info *model)
{
    if (model == NULL) return;

    release_matrix(model->beta);
    release_vector(model->lambda);
    release_vector(model->a0);
    release_vector(model->rsqr);

    free(model->n_zero);

    release_vector(model->xm);
    release_vector(model->xs);

    release_vector(model->cv_mse);
    release_vector(model->cv_se);

    free(model);
}



struct elasticnet_info *elasticnet_load(FILE *src)
{
    int i, j;
    int p, K;
    double delta;

    struct elasticnet_info *model;

    fscanf(src, "%d %d", &p, &K);

    model = create_empty_model(p, K);

    fscanf(src, "%lf", &(model->ym));
    for (j = 0; j < model->p; ++j) fscanf(src,  "%lf", &(model->xm[j]));

    fscanf(src, "%lf", &(model->ys));
    for (j = 0; j < model->p; ++j) fscanf(src,  "%lf", &(model->xs[j]));

    fscanf(src, "%d", &(model->n_folds));

    if (model->n_folds > 0) {
        model->cv_mse = new_vector(model->K);
        model->cv_se  = new_vector(model->K);
        for (i = 0; i < model->K; ++i) {
            fscanf(src, "%lf %lf %lf %lf %d %lf",
                   &(model->lambda[i]), &(model->cv_mse[i]), &(model->cv_se[i]),
                   &(model->rsqr[i]), &(model->n_zero[i]), &(model->a0[i]));
            for (j = 0; j < model->p; ++j) fscanf(src, "%lf", &(model->beta[i][j]));
        }
        model->cv_mse_min = 0;
        for (i = 1; i < model->K; ++i) if (model->cv_mse[i] < model->cv_mse[model->cv_mse_min]) model->cv_mse_min = i;
        delta = model->cv_mse[model->cv_mse_min] + model->cv_se[model->cv_mse_min];
        model->cv_mse_1se = model->cv_mse_min;
        for (i = model->cv_mse_1se; i > 0; --i) if (model->cv_mse[i] < delta) model->cv_mse_1se = i;
    } else {
        for (i = 0; i < model->K; ++i) {
            fscanf(src, "%lf %lf %d %lf", &(model->lambda[i]), &(model->rsqr[i]), &(model->n_zero[i]), &(model->a0[i]));
            for (j = 0; j < model->p; ++j) fscanf(src, "%lf", &(model->beta[i][j]));
        }
    }
    return model;
}



void elasticnet_write(struct elasticnet_info *model, FILE *dest)
{
    int i, j;

    fprintf(dest, "%d %d\n", model->p, model->K);

    fprintf(dest, "%f", model->ym);
    for (j = 0; j < model->p; ++j) fprintf(dest,  " %f", model->xm[j]);
    fprintf(dest, "\n");

    fprintf(dest, "%f", model->ys);
    for (j = 0; j < model->p; ++j) fprintf(dest,  " %f", model->xs[j]);
    fprintf(dest, "\n");

    fprintf(dest, "%d\n", model->n_folds);
    if (model->n_folds > 0) {
        for (i = 0; i < model->K; ++i) {
            fprintf(dest, "%f %f %f %f %d %f",
                    model->lambda[i], model->cv_mse[i], model->cv_se[i],
                    model->rsqr[i], model->n_zero[i], model->a0[i]);
            for (j = 0; j < model->p; ++j) fprintf(dest, " %f", model->beta[i][j]);
            fprintf(dest, "\n");
        }
    } else {
        for (i = 0; i < model->K; ++i) {
            fprintf(dest, "%f %f %d %f", model->lambda[i], model->rsqr[i], model->n_zero[i], model->a0[i]);
            for (j = 0; j < model->p; ++j) fprintf(dest, " %f", model->beta[i][j]);
            fprintf(dest, "\n");
        }
    }
}



void elasticnet_coefficients(struct elasticnet_info *model, double lambda, double *beta, double *a0)
{
    int i, j, l;
    double frac;
    bool interpolate;

    l = -1;
    interpolate = false;
    if (isnan(lambda)) {
        if (model->n_folds > 1) {
            /* if cross-validation was used to build the model, find
             * coefficients associated with highest lambda that produces a
             * CV MSE less than the lowest overall MSE + 1 SE (the "one
             * standard error" rule) */
            l = model->cv_mse_1se;
        } else {
            /* otherwise, use model associated with highest R^2 */
            l = 0;
            for (i = 1; i < model->K; ++i) if (model->rsqr[l] < model->rsqr[i]) l = i;
        }
    } else if ((lambda - model->lambda[0]) > LAMBDA_TOLERANCE) {
        fprintf(stderr, "Warning: chosen lambda %f is greater than that used in model "
                        "fitting - substituting with largest fitted lambda %f.\n",
                lambda, model->lambda[0]);
        l = 0;
    } else if ((model->lambda[model->K - 1] - lambda) > LAMBDA_TOLERANCE) {
        fprintf(stderr, "Warning: chosen lambda %f is smaller than that used in model "
                        "fitting - substituting with smallest fitted lambda %f.\n",
                lambda, model->lambda[model->K - 1]);
        l = model->K - 1;
    } else {
        for (i = 0; i < model->K; ++i) {
            if (fabs(lambda - model->lambda[i]) < LAMBDA_TOLERANCE) {
                l = i;
                break;
            }
        }

        if (l < 0) {
            for (i = 1; i < model->K; ++i) {
                if (lambda < model->lambda[i - 1] && lambda > model->lambda[i]) {
                    l = i;
                    interpolate = true;
                    break;
                }
            }
        }
    }

    if (interpolate) {
        fprintf(stderr, "Warning: model fitting did not include lambda %f, so resorting to linear "
                        "interpolation between coefficients for lambda = %f and %f.\n",
                lambda, model->lambda[l - 1], model->lambda[l]);

        frac = ((lambda - model->lambda[l - 1]) / (model->lambda[l - 1] - model->lambda[l - 1]));
        fprintf(stderr, "Fraction of interpolation: %f\n", frac);
        for (j = 0; j < model->p; ++j) {
            beta[j] = model->beta[l - 1][j] + (model->beta[l][j] - model->beta[l - 1][j]) * frac;
        }

        *a0 = model->ym;
        for (j = 0; j < model->p; ++j) *a0 -= beta[j] * model->xm[j];
    } else {
        memcpy(beta, model->beta[l], model->p * sizeof(double));
        *a0 = model->a0[l];
    }
}
