### Langevin Analysis in One and Two Dimensions

The [Langevin package](http://cran.r-project.org/package=Langevin) provides R
functions to estimate drift and diffusion functions from time series and
generate synthetic time series from given drift and diffusion coefficients.


### Documentation

All functions of the Langevin package have corresponding help files.
Additionally the package ships a pdf vignette which corresponds to a paper on
[arXiv](http://arxiv.org/pdf/1603.02036).


### Citation

To cite the Langevin package and/or the mathematical concept behind it correctly
see 'citation("Langevin")' for details.


### Examples

The help files for each function contain usage examples, additionally the
package repository ships a
[script](https://gitlab.uni-oldenburg.de/TWiSt/Langevin/raw/master/examples.r)
with examples that reproduce the figures from the vignette.


### Installation

Released and tested versions of the Langevin package are available at
[CRAN](http://cran.r-project.org) and can be installed from within R via

```R
install.packages("Langevin")
```

The development version of the Langevin package can be installed from within R
via

```R
install.packages("devtools")
devtools::install_git("https://gitlab.uni-oldenburg.de/TWiSt/Langevin.git")
```


### Authors

Philip Rinn, Pedro G. Lind and David Bastine


### License

GPL (>= 2)
