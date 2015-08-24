#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <math.h>
#include <string.h>

#include <sys/time.h>

#include "mt19937ar.h"

#include "elasticnet.h"

#include "data.h"

struct data_set_details {
    int n_features;

    double **train_X;
    double  *train_Y;
    int      n_train;

    double **test_X;
    double  *test_Y;
    int      n_test;
};


static void shuffle_ints(int *a, int n, double (*rnd)(void))
{
    int i, j, t;

    for (i = n - 1; i > 0; --i) {
        j = (int)(rnd() * i);
        t = a[i]; a[i] = a[j]; a[j] = t;
    }
}


int main(int argc, char **argv)
{
    struct data_set_details details;
    struct elasticnet_info *elnet;
    FILE *model_file;

    int i, *fold_ids, n_folds;
    double *yhat, rmse_1se, rmse_min;

    struct timeval t;
    gettimeofday(&t, NULL);
    init_genrand(t.tv_usec);

    load_fold(argv[1], argv[2], atoi(argv[3]),
              &(details.train_X), &(details.train_Y), &(details.n_train),
              &(details.test_X), &(details.test_Y), &(details.n_test),
              &(details.n_features));

    n_folds = 10;
    fold_ids = malloc(details.n_train * sizeof(int));
    for (i = 0; i < details.n_train; ++i) fold_ids[i] = i % n_folds;
    if (argc > 5) shuffle_ints(fold_ids, details.n_train, genrand_real2);

    elnet = elasticnet_cv_fit(details.train_X, details.train_Y, details.n_train, details.n_features,
                              fold_ids, n_folds, NULL, 100, atof(argv[4]));

    free(fold_ids);

    model_file = fopen("elnet_model.txt", "w");
    elasticnet_write(elnet, model_file);
    fclose(model_file);

    yhat = malloc(details.n_test * sizeof(double));

    elasticnet_predict(elnet, details.test_X, details.n_test, NAN, yhat);

    rmse_1se = 0;
    for (i = 0; i < details.n_test; ++i) {
        rmse_1se += ((details.test_Y[i] - yhat[i]) * (details.test_Y[i] - yhat[i]) - rmse_1se) / (i + 1);
    }
    rmse_1se = sqrt(rmse_1se);

    elasticnet_predict(elnet, details.test_X, details.n_test, elnet->lambda[elnet->cv_mse_min], yhat);

    rmse_min = 0;
    for (i = 0; i < details.n_test; ++i) {
        rmse_min += ((details.test_Y[i] - yhat[i]) * (details.test_Y[i] - yhat[i]) - rmse_min) / (i + 1);
    }
    rmse_min = sqrt(rmse_min);

    fprintf(stdout, "%f %f %f %f\n",
            rmse_min, elnet->lambda[elnet->cv_mse_min],
            rmse_1se, elnet->lambda[elnet->cv_mse_1se]);

    free(yhat);

    elasticnet_free(elnet);

    unload_data(details.train_X, details.train_Y, details.test_X, details.test_Y);

    return EXIT_SUCCESS;
}
