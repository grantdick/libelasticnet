#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <string.h> /* needed for string manipulation and memcpy */

#include "data.h"
#include "readline.h"

static double **read_values(char *source_file, int *n_instances, int *n_values)
{
    int i;

    int n_rows, n_cols, n_val;
    double **data;
    double *values;

    FILE *input;
    char *buffer, *line, *tok;
    size_t bufsz;

    n_val = 10000;
    values = malloc(n_val * sizeof(double));

    input = fopen(source_file, "r");
    buffer = line = tok = NULL;
    bufsz = 0;

    n_rows = n_cols = 0;
    line = next_line(&buffer, &bufsz, input);
    while (!feof(input)) {
        if (strlen(line) > 0) {
            for (i = 0, tok = strtok(line, " \t"); tok; i++, tok = strtok(NULL, " \t")) {
                if ((n_rows * n_cols + i + 1) >= n_val) {
                    n_val *= 2;
                    values = realloc(values, n_val * sizeof(double));
                }
                values[n_rows * n_cols + i] = atof(tok);
            }

            if (n_cols == 0) n_cols = i;
            n_rows++;
        }
        line = next_line(&buffer, &bufsz, input);
    }
    fclose(input);
    free(buffer);

    data = malloc(n_rows * sizeof(double *));
    for (i = 0; i < n_rows; ++i) data[i] = values + i * n_cols;

    *n_instances = n_rows;
    *n_values = n_cols;

    return data;
}

static void load_folds(char *fold_file, int fold, int n_instances,
                       int **train_instances, int *n_train,
                       int **test_instances,  int *n_test)
{
    int i, j;

    FILE *input;
    char *buffer, *line, *tok;
    size_t bufsz;

    int *train, *test;
    bool in_test;

    train = malloc(n_instances * sizeof(int));
    test  = malloc(n_instances * sizeof(int));

    input = fopen(fold_file, "r");
    buffer = NULL;
    bufsz = 0;
    for (line = NULL, i = 0; i < fold; ++i) line = next_line(&buffer, &bufsz, input);
    fclose(input);

    *n_train = *n_test = 0;
    for (tok = strtok(line, "\t"); tok; tok = strtok(NULL, "\t")) {
        test[(*n_test)++] = atoi(tok) - 1;
    }
    free(buffer);
    buffer = line = tok = NULL;
    bufsz = 0;

    for (i = 0; i < n_instances; ++i) {
        for (in_test = false, j = 0; j < *n_test && !in_test; ++j) in_test = (i == test[j]);
        if (!in_test) train[(*n_train)++] = i;
    }

    *train_instances = train;
    *test_instances = test;
}

void load_fold(char *source_file, char *fold_file, int fold,
               double ***trainX, double **trainY, int *n_train,
               double ***testX, double **testY, int *n_test,
               int *n_features)
{
    int i;
    double **all_data, **X, *Y;
    int n_rows, n_cols, n_feat;
    int *train_instances, *test_instances;

    all_data = read_values(source_file, &n_rows, &n_cols);
    n_feat = n_cols - 1;
    *n_features = n_feat;

    load_folds(fold_file, fold, n_rows, &train_instances, n_train, &test_instances, n_test);

    X = malloc(*n_train * sizeof(double *));
    X[0] = malloc(*n_train * n_feat * sizeof(double));
    Y = malloc(*n_train * sizeof(double));
    for (i = 0; i < *n_train; ++i) {
        X[i] = X[0] + (i * n_feat);
        memcpy(X[i], all_data[train_instances[i]], n_feat * sizeof(double));
        Y[i] = all_data[train_instances[i]][n_feat];
    }
    *trainX = X;
    *trainY = Y;

    X = malloc(*n_test * sizeof(double *));
    X[0] = malloc(*n_test * n_feat * sizeof(double));
    Y = malloc(*n_test * sizeof(double));
    for (i = 0; i < *n_test; ++i) {
        X[i] = X[0] + (i * n_feat);
        memcpy(X[i], all_data[test_instances[i]], n_feat * sizeof(double));
        Y[i] = all_data[test_instances[i]][n_feat];
    }
    *testX = X;
    *testY = Y;

    free(test_instances);
    free(train_instances);
    free(all_data[0]);
    free(all_data);
}



void unload_data(double **trainX, double *trainY, double **testX, double *testY)
{
    if (trainX) free(trainX[0]);
    free(trainX);
    free(trainY);

    if (testX) free(testX[0]);
    free(testX);
    free(testY);
}
