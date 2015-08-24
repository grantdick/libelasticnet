libelasticnet: Simple Elastic Net in C
========================================================================

This is a very simple library for performing penalised linear
regression via the elastic net. It is intended primarily as a library
for learning how the method works, and for embedding in projects where
a very simple elastic net implementation is required. The library
offers the following features:

* cross-validation of training data to estimate hyperparameters for
  the regression process (both min. CVMSE and the "one standard error"
  rule are implemented)
* automatic computation of the sequence of values for lambda for the
  regularisation path (or user-supplied, if desired), and warm starts
  along the path
* loading and saving models to and from FILE streams

This code was written primarily for some internal projects that
required a simple lasso implementation. It's been quite useful, both
as a learning exercise, and also as a quick and convenient solution
when required. However, it should be stressed - for anything
"serious", you should use a real package (e.g.,
[glmnet](https://cran.r-project.org/web/packages/glmnet/index.html))!
This code makes some efforts towards optimisation (mainly through the
tricks suggested in the paper below, but the main goal was to keep it
simple and understandable (and reasonably clean, code-wise).

For more information on the coordinate descent method for the elastic
net, the following paper is recommended:<br />
Friedman, J., Hastie, T. and Tibshirani, R. (2010) _Regularization
Paths for Generalized Linear Models via Coordinate Descent_, Journal
of Statistical Software, Vol. 33(1)
[link](http://www.jstatsoft.org/v33/i01/)

This code borrows some ideas from the
[glmnet](https://cran.r-project.org/web/packages/glmnet/index.html)
Fortran source code, and [lasso4j](https://github.com/yasserg/lasso4j)
project, so thanks to the authors of those packages for making the
source available!

Installation
------------

This project should install on any platform supported by GCC. A
Makefile is included to build the base library and demo application. A
call to:
  make
should compile the project without any problems, and the resulting
binaries should appear in the dist directory.

Documentation
-------------

Documentation for the project can be found in the file [doc.txt](doc.txt)

License
-------

See [LICENSE](LICENSE) file.

Support
-------

Any questions or comments should be directed to Grant Dick
(grant.dick@otago.ac.nz)
