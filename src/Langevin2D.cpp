// Copyright (C) 2012 - 2015  Philip Rinn
// Copyright (C) 2012 - 2015  Carl von Ossietzy Universität Oldenburg
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, see <https://www.gnu.org/licenses/gpl-2.0>.

// [[Rcpp::depends(RcppArmadillo)]]
#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include "linreg.h"
#ifdef _OPENMP
    #include <omp.h>
#endif
using namespace Rcpp;

//' Calculate the Drift and Diffusion of two-dimensional stochastic processes
//'
//' \code{Langevin2D} calculates the Drift (with error) and Diffusion matrices
//' for given time series.
//'
//'
//' @param data a matrix containing the time series as rows.
//' @param bins a scalar denoting the number of \code{bins} to calculate Drift
//' and Diffusion on.
//' @param steps a vector giving the \eqn{\tau} steps to calculate the moments
//' (in samples).
//' @param sf a scalar denoting the sampling frequency.
//' @param bin_min a scalar denoting the minimal number of events per \code{bin}.
//' Defaults to \code{100}.
//' @param reqThreads a scalar denoting how many threads to use. Defaults to
//' \code{-1} which means all available cores.
//'
//' @return \code{Langevin2D} returns a list with nine components:
//' @return \item{D1}{a tensor with all values of the drift coefficient.
//' Dimension is \code{bins} x \code{bins} x 2. The first
//' \code{bins} x \code{bins} elements define the drift \eqn{D^{(1)}_{1}}
//' for the first variable and the rest define the drift \eqn{D^{(1)}_{2}}
//' for the second variable.}
//' @return \item{eD1}{a tensor with all estimated errors of the drift
//' coefficient. Dimension is \code{bins} x \code{bins} x 2. Same expression as
//' above.}
//' @return \item{D2}{a tensor with all values of the diffusion coefficient.
//' Dimension is \code{bins} x \code{bins} x 3. The first
//' \code{bins} x \code{bins} elements define the diffusion \eqn{D^{(2)}_{11}},
//' the second \code{bins} x \code{bins} elements define the diffusion
//' \eqn{D^{(2)}_{22}} and the rest define the diffusion
//' \eqn{D^{(2)}_{12} = D^{(2)}_{21}}.}
//' @return \item{eD2}{a tensor with all estimated errors of the driffusion
//' coefficient. Dimension is \code{bins} x \code{bins} x 3. Same expression as
//' above.}
//' @return \item{mean_bin}{a matrix of the mean value per \code{bin}.
//' Dimension is \code{bins} x \code{bins} x 2. The first
//' \code{bins} x \code{bins} elements define the mean for the first variable
//' and the rest for the second variable.}
//' @return \item{density}{a matrix of the number of events per \code{bin}.
//' Rows label the \code{bin} of the first variable and columns the second
//' variable.}
//' @return \item{M1}{a tensor of the first moment for each \code{bin} (line
//' label) and  each \eqn{\tau} step (column label). Dimension is
//' \code{bins} x \code{bins} x 2\code{length(steps)}.}
//' @return \item{eM1}{a tensor of the standard deviation of the first
//' moment for each bin (line label) and  each \eqn{\tau} step (column label).
//' Dimension is \code{bins} x \code{bins} x 2\code{length(steps)}.}
//' @return \item{M2}{a tensor of the second moment for each bin (line
//' label) and  each \eqn{\tau} step (column label). Dimension is
//' \code{bins} x \code{bins} x 3\code{length(steps)}.}
//' @return \item{U}{a matrix of the \code{bin} borders}
//'
//' @author Philip Rinn
//' @seealso \code{\link{Langevin1D}}
//' @import Rcpp
//' @useDynLib Langevin
//' @export
// [[Rcpp::export]]
List Langevin2D(const arma::mat& data, const int& bins, const arma::vec& steps,
                const double& sf=1, const int& bin_min=100, int reqThreads=-1) {
    // Set the number of threads
    #ifdef _OPENMP
        int haveCores = omp_get_num_procs();
        if(reqThreads <= 0 || reqThreads > haveCores)
            reqThreads = haveCores;
        omp_set_num_threads(reqThreads);
    #endif
    int nsteps = steps.n_elem;
    arma::mat U(2,(bins+1));
    U.row(0) = arma::linspace<arma::rowvec>(arma::min(data.row(0)),arma::max(data.row(0)),(bins+1));
    U.row(1) = arma::linspace<arma::rowvec>(arma::min(data.row(1)),arma::max(data.row(1)),(bins+1));
    arma::cube M1(bins, bins, 2*nsteps);
    arma::cube eM1(bins, bins, 2*nsteps);
    arma::cube M2(bins, bins, 3*nsteps);
    arma::cube D1(bins, bins, 2);
    arma::cube eD1(bins, bins, 2);
    arma::cube D2(bins, bins, 3);
    arma::mat dens(bins, bins);
    arma::cube mean_bin(bins, bins, 2);
    M1.fill(NA_REAL);
    eM1.fill(NA_REAL);
    M2.fill(NA_REAL);
    D1.fill(NA_REAL);
    eD1.fill(NA_REAL);
    D2.fill(NA_REAL);
    mean_bin.zeros();

#pragma omp parallel default(shared)
{
    #pragma omp for collapse(2)
    for (int i = 0; i < bins; i++) {
        for (int j = 0; j < bins; j++) {
            arma::mat sum_m1(2, nsteps);
            arma::mat sum_m2(3, nsteps);
            arma::vec len_step(nsteps);
            sum_m1.zeros();
            sum_m2.zeros();
            len_step.zeros();
            double len_bin = 0;;
            for (int n = 0; n < data.n_cols - steps.max(); n++) {
                if(data(0,n) >= U(0,i) && data(0,n) < U(0,i+1) && data(1,n) >= U(1,j) && data(1,n) < U(1,j+1) && arma::is_finite(data(0,n)) && arma::is_finite(data(1,n))) {
                    for (int s = 0; s < nsteps; s++) {
                        if(arma::is_finite(data(0,n+steps(s))) && arma::is_finite(data(1,n+steps(s)))) {
                            double inc0 = data(0,n+steps(s)) - data(0,n);
                            double inc1 = data(1,n+steps(s)) - data(1,n);
                            sum_m1(0,s) += inc0;
                            sum_m1(1,s) += inc1;
                            sum_m2(0,s) += inc0*inc0;
                            sum_m2(1,s) += inc0*inc1;
                            sum_m2(2,s) += inc1*inc1;
                            len_step(s)++;
                        }
                    }
                    mean_bin(i,j,0) += data(0,n);
                    mean_bin(i,j,1) += data(1,n);
                    len_bin++;
                }
            }
            mean_bin(i,j,0) /= len_bin;
            mean_bin(i,j,1) /= len_bin;
            dens(i,j) = arma::max(len_step);
            if(len_bin >= bin_min) {
                M1(arma::span(i),arma::span(j),arma::span(0,nsteps-1)) = sum_m1.row(0)/arma::trans(len_step);          // dim1
                M1(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1)) = sum_m1.row(1)/arma::trans(len_step);   // dim2
                M2(arma::span(i),arma::span(j),arma::span(0,nsteps-1)) = sum_m2.row(0)/arma::trans(len_step);          // M2_11
                M2(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1)) = sum_m2.row(1)/arma::trans(len_step);   // M2_12
                M2(arma::span(i),arma::span(j),arma::span(2*nsteps,3*nsteps-1)) = sum_m2.row(2)/arma::trans(len_step); // M2_22

                eM1(arma::span(i),arma::span(j),arma::span(0,nsteps-1)) = arma::sqrt((M2(arma::span(i),arma::span(j),arma::span(0,nsteps-1)) - arma::square(M1(arma::span(i),arma::span(j),arma::span(0,nsteps-1)))));
                eM1(arma::span(i),arma::span(j),arma::span(0,nsteps-1)) /= arma::sqrt(len_step);
                eM1(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1)) = arma::sqrt((M2(arma::span(i),arma::span(j),arma::span(2*nsteps,3*nsteps-1)) - arma::square(M1(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1)))));
                eM1(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1)) /= arma::sqrt(len_step);

                // linear regression with weights to get D1 and D2
                // D1
                arma::vec y = M1(arma::span(i),arma::span(j),arma::span(0,nsteps-1));
                arma::vec w = eM1(arma::span(i),arma::span(j),arma::span(0,nsteps-1));
                arma::vec coef = linreg(steps, y, 1/w);
                D1(i,j,0) = sf*coef(1);
                y = M1(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1));
                w = eM1(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1));
                coef = linreg(steps, y, 1/w);
                D1(i,j,1) = sf*coef(1);
                // D2
                // M2 - (tau * D1)^2 = tau * D2
                y = M2(arma::span(i),arma::span(j),arma::span(0,nsteps-1));
                y -= arma::square(steps*D1(i,j,0)/sf);
                coef = linreg(steps, y);
                D2(i,j,0) = sf*coef(1)/2;
                y = M2(arma::span(i),arma::span(j),arma::span(nsteps,2*nsteps-1));
                y -= arma::square(steps/sf)*D1(i,j,0)*D1(i,j,1);
                coef = linreg(steps, y);
                D2(i,j,1) = sf*coef(1)/2;
                y = M2(arma::span(i),arma::span(j),arma::span(2*nsteps,3*nsteps-1));
                y -= arma::square(steps*D1(i,j,1)/sf);
                coef = linreg(steps, y);
                D2(i,j,2) = sf*coef(1)/2;
                // calculate the error of D1
                eD1(i,j,0) = sqrt(sf*D2(i,j,0)/dens(i,j) - D1(i,j,0)*D1(i,j,0)/dens(i,j));
                eD1(i,j,1) = sqrt(sf*D2(i,j,2)/dens(i,j) - D1(i,j,1)*D1(i,j,1)/dens(i,j));
            }
        }
    }
}
    List ret;
    ret["D1"] = D1;
    ret["eD1"] = eD1;
    ret["D2"] = D2;
    ret["mean_bin"] = mean_bin;
    ret["density"] = dens;
    ret["M1"] = M1;
    ret["eM1"] = eM1;
    ret["M2"] = M2;
    ret["U"] = U;
    return ret;
}
