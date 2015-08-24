#ifndef _ELASTICNET_H
#define	_ELASTICNET_H

#ifdef	__cplusplus
extern "C" {
#endif

    #include <stdio.h>
    #include <stdbool.h>

    /* contains the relevant model information so that prediction and
     * reporting can take place */
    struct elasticnet_info {
        int p;          /* number of predictors */

        int K;          /* the number of penalty coefficients used
                         * during the model fit */

        double *lambda; /* the norm penalty weights used in model
                         * fitting */

        double *a0;     /* the model intercept for a given value of
                         * lambda */

        double **beta;  /* the compressed coefficients for a given
                         * value of lambda */

        double *rsqr;   /* R^2 values observed at each fit of
                         * lambda */

        int *n_zero;    /* the number of non-zero predictors for a
                         * given lambda */

        double *xm;     /* unscaled mean of predictors */
        double *xs;     /* unscaled std. deviation of predictors */

        double ym;      /* unscaled mean of response */
        double ys;      /* unscaled std. deviation of response */

        /* if cross-validation is used during model building, then
         * these fields will contain the relevant cross-validtion
         * information. Otherwise, these fields will be NULL */
        int n_folds;     /* number of folds used */
        double *cv_mse;  /* cross-validation mean-squared error */
        double *cv_se;   /* cross-validation standard error */

        int cv_mse_min;  /* index of lambda that minimises cv MSE */
        int cv_mse_1se;  /* index of lambda that minimised cv MSE + 1SE */
    };

    /* computes the regularisation path and then uses coordinate
     * descent to fit regularised linear models along this
     * path. Cross-validation is then used to select the points along
     * that path that result in the lowest mean squared error (both
     * over all, and within one standard error of the lowest)
     *
     * ARGUMENTS:
     *
     * double **X     - the input features of the problem, these will be
     *                  copied internally and then standardised to have
     *                  zero mean and unit variance prior to fitting
     *
     * double *y      - the vector of response values, these will also be
     *                  copied and standardised before fitting
     *
     * int N, p       - the number of instances and features, respectively
     *
     * int *fold_id   - an array (length N) that contains the fold id for
     *                  each instance in the data set, if this is NULL,
     *                  then a simple (non-random) 1..n_folds sequence is
     *                  created for the instances in the data set (i.e.,
     *                  [0, 1, ..., n_folds - 1, 0, 1, ..., n_folds - 1, ...]
     *                  until the array is full)
     *
     * int n_folds    - the number of folds used in cross-validation
     *
     * double *lambda - the points along the regularisation path for model
     *                  fitting. If this is NULL, then this is computed
     *                  automatically according to the rules in Friedman et
     *                  al. (2010)
     *
     * int K          - the number of points along the regularisation path
     *
     * double alpha   - the compromise term between ridge (l2) and lasso (l1)
     *                  penalties. alpha=0 is a full ridge-regression, while
     *                  alpha=1 results in a lasso model. When p >> N, this
     *                  intermediate values for alpha may be useful to group
     *                  related features
     *
     * RETURNS: a structure containing the relevant model fitting
     * information (coefficients, cross-validation errors, and so on)
     */
    struct elasticnet_info *elasticnet_cv_fit(double **X, double *y, int N, int p,
                                              int *fold_id, int n_folds,
                                              double *lambda, int K, double alpha);

    /* computes the regularisation path and then uses coordinate
     * descent to fit regularised linear models along this
     * path.
     *
     * ARGUMENTS:
     *
     * double **X     - the input features of the problem, these will be
     *                  copied internally and then standardised to have
     *                  zero mean and unit variance prior to fitting
     *
     * double *y      - the vector of response values, these will also be
     *                  copied and standardised before fitting
     *
     * int N, p       - the number of instances and features, respectively
     *
     * double *lambda - the points along the regularisation path for model
     *                  fitting. If this is NULL, then this is computed
     *                  automatically according to the rules in Friedman et
     *                  al. (2010)
     *
     * int K          - the number of points along the regularisation path
     *
     * double alpha   - the compromise term between ridge (l2) and lasso (l1)
     *                  penalties. alpha=0 is a full ridge-regression, while
     *                  alpha=1 results in a lasso model. When p >> N, this
     *                  intermediate values for alpha may be useful to group
     *                  related features
     *
     * RETURNS: a structure containing the relevant model fitting
     * information WITHOUT any cross-validation to estimate suitable
     * parameter settings
     */
    struct elasticnet_info *elasticnet_fit(double **X, double *y, int N, int p,
                                           double *lambda, int K, double alpha);

    /* uses the supplied model structure to predict unknown response
     * values for the supplied data set X, at a point along the
     * regularisation path specified by lambda
     *
     * ARGUMENTS:
     *
     * struct model   - a elastic net model information structure, fitted
     *                  through elasticnet_fit or elasticnet_cv_fit, or loaded
     *                  through elasticnet_load
     *
     * double **X     - the input features of the problem, these will be
     *                  copied internally and then standardised to have
     *                  zero mean and unit variance prior to fitting
     *
     * int N          - the number of instances in X
     *
     * double lambda  - the points along the regularisation path for model
     *                  fitting. If this is NULL, then this is computed
     *                  automatically according to the rules in Friedman et
     *                  al. (2010)
     *
     * double *yhat   - the vector of response prediction values, this must
     *                  be allocated to the right size BEFORE being supplied
     *                  as an argument (i.e., no memory allocation takes place
     *                  in the function)
     *
     * RETURNS: nothing - predictions are placed in the parameter yhat
     */
    void elasticnet_predict(struct elasticnet_info *model, double **X, int N, double lambda, double *yhat);

    /* uses the supplied model structure to predict unknown response
     * values for the supplied data set X, at a point along the
     * regularisation path specified by lambda
     *
     * ARGUMENTS:
     *
     * struct model   - a elastic net model information structure, fitted
     *                  through elasticnet_fit or elasticnet_cv_fit, or loaded
     *                  through elasticnet_load
     *
     */
    void elasticnet_free(struct elasticnet_info *model);

    /* uses the supplied file to load a previously built elastic net
     * model. TODO: the file specification is currently undocumented,
     * but fairly straightforward to interpret
     *
     * ARGUMENTS:
     *
     * FILE src       - the file that contains the model spec
     *
     * RETURNS: the model structure that corresponds to the model spec
     * in the file
     */
    struct elasticnet_info *elasticnet_load(FILE *src);

    /* writes the provided model structure to the specified file
     * stream using a simple file format suitable for reloading
     * through elasticnet_load()
     *
     * ARGUMENTS:
     *
     * struct model   - a elastic net model information structure, fitted
     *                  through elasticnet_fit or elasticnet_cv_fit, or loaded
     *                  through elasticnet_load
     *
     * FILE dest      - the file to which the model spec is written
     */
    void elasticnet_write(struct elasticnet_info *model, FILE *dest);

    /* uses the supplied model structure to predict unknown response
     * values for the supplied data set X, at a point along the
     * regularisation path specified by lambda
     *
     * ARGUMENTS:
     *
     * struct model   - a elastic net model information structure, fitted
     *                  through elasticnet_fit or elasticnet_cv_fit, or loaded
     *                  through elasticnet_load
     *
     * double lambda  - the points along the regularisation path for model
     *                  fitting. If this is NULL, then this is computed
     *                  automatically according to the rules in Friedman et
     *                  al. (2010)
     *
     * double *beta   - the vector of response prediction values, this must
     *                  be allocated to the right size BEFORE being supplied
     *                  as an argument (i.e., no memory allocation takes place
     *                  in the function)
     *
     * double *a0     - pointer to the variable that will take on the value of
     *                  the model intercept
     *
     * RETURNS: nothing - coefficients are placed in the parameters beta and a0
     */
    void elasticnet_coefficients(struct elasticnet_info *model, double lambda, double *beta, double *a0);

#ifdef	__cplusplus
}
#endif

#endif
