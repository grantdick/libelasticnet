#ifndef _ELNET_UTIL_H
#define	_ELNET_UTIL_H

#ifdef	__cplusplus
extern "C" {
#endif

    void standardise_data(double **x, double *y, int N, int p,
                          double *xm, double *xs,
                          double *ym, double *ys);

    double **new_matrix(int n, int m);
    double **copy_matrix(double **x, int n, int m);
    void     release_matrix(double **x);

    double *new_vector(int n);
    double *copy_vector(double *x, int n);
    void    release_vector(double *x);

#ifdef	__cplusplus
}
#endif

#endif
