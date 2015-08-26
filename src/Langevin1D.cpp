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

//' Calculate the Drift and Diffusion of one-dimensional stochastic processes
//'
//' \code{Langevin1D} calculates the Drift and Diffusion vectors (with errors)
//' for a given time series.
//'
//'
//' @param data a vector containing the time series.
//' @param bins a scalar denoting the number of \code{bins} to calculate the
//' conditional moments on.
//' @param steps a vector giving the \eqn{\tau} steps to calculate the
//' conditional moments (in samples (=\eqn{\tau * sf})).
//' @param sf a scalar denoting the sampling frequency.
//' @param bin_min a scalar denoting the minimal number of events per \code{bin}.
//' Defaults to \code{100}.
//' @param reqThreads a scalar denoting how many threads to use. Defaults to
//' \code{-1} which means all available cores.
//'
//' @return \code{Langevin1D} returns a list with thirteen components:
//' @return \item{D1}{a vector of the Drift coefficient for each \code{bin}.}
//' @return \item{eD1}{a vector of the error of the Drift coefficient for each
//' \code{bin}.}
//' @return \item{D2}{a vector of the Diffusion coefficient for each \code{bin}.}
//' @return \item{eD2}{a vector of the error of the Driffusion coefficient for
//' each \code{bin}.}
//' @return \item{D4}{a vector of the fourth Kramers-Moyal coefficient for each
//' \code{bin}.}
//' @return \item{mean_bin}{a vector of the mean value per \code{bin}.}
//' @return \item{density}{a vector of the number of events per \code{bin}.}
//' @return \item{M1}{a matrix of the first conditional moment for each
//' \eqn{\tau}. Rows corespond to \code{bin}, columns to \eqn{\tau}.}
//' @return \item{eM1}{a matrix of the error of the first conditional moment
//' for each \eqn{\tau}. Rows corespond to \code{bin}, columns to \eqn{\tau}.}
//' @return \item{M2}{a matrix of the second conditional moment for each
//' \eqn{\tau}. Rows corespond to \code{bin}, columns to \eqn{\tau}.}
//' @return \item{eM2}{a matrix of the error of the second conditional moment
//' for each \eqn{\tau}. Rows corespond to \code{bin}, columns to \eqn{\tau}.}
//' @return \item{M4}{a matrix of the fourth conditional moment for each
//' \eqn{\tau}. Rows corespond to \code{bin}, columns to \eqn{\tau}.}
//' @return \item{U}{a vector of the \code{bin} borders.}
//'
//' @author Philip Rinn
//' @seealso \code{\link{Langevin2D}}
//' @examples
//'
//' # Set number of bins, steps and the sampling frequency
//' bins <- 20;
//' steps <- c(1:5);
//' sf <- 1000;
//'
//' #### Linear drift, constant diffusion
//'
//' # Generate a time series with linear D^1 = -x and constant D^2 = 1
//' x <- timeseries1D(N=1e6, d11=-1, d20=1, sf=sf);
//' # Do the analysis
//' est <- Langevin1D(x, bins, steps, sf, reqThreads=2);
//' # Plot the result and add the theoretical expectation as red line
//' plot(est$mean_bin, est$D1);
//' lines(est$mean_bin, -est$mean_bin, col='red');
//' plot(est$mean_bin, est$D2);
//' abline(h=1, col='red');
//'
//' #### Cubic drift, constant diffusion
//'
//' # Generate a time series with cubic D^1 = x - x^3 and constant D^2 = 1
//' x <- timeseries1D(N=1e6, d13=-1, d11=1, d20=1, sf=sf);
//' # Do the analysis
//' est <- Langevin1D(x, bins, steps, sf, reqThreads=2);
//' # Plot the result and add the theoretical expectation as red line
//' plot(est$mean_bin, est$D1);
//' lines(est$mean_bin, est$mean_bin - est$mean_bin^3, col='red');
//' plot(est$mean_bin, est$D2);
//' abline(h=1, col='red');
//' @import Rcpp
//' @useDynLib Langevin
//' @export
// [[Rcpp::export]]
List Langevin1D(const arma::vec& data, const int& bins, const arma::vec& steps,
                const double& sf=1, const int& bin_min=100, int reqThreads=-1) {
    // Set the number of threads
    #ifdef _OPENMP
        int haveCores = omp_get_num_procs();
        if(reqThreads <= 0 || reqThreads > haveCores)
            reqThreads = haveCores;
        omp_set_num_threads(reqThreads);
    #endif
    arma::vec U = arma::linspace<arma::vec>(data.min(), data.max(), (bins+1));
    int nsteps = steps.n_elem;
    arma::mat M1(bins, nsteps);
    arma::mat eM1(bins, nsteps);
    arma::mat M2(bins, nsteps);
    arma::mat eM2(bins, nsteps);
    arma::mat M4(bins, nsteps);
    arma::vec D1(bins);
    arma::vec eD1(bins);
    arma::vec D2(bins);
    arma::vec eD2(bins);
    arma::vec D4(bins);
    arma::vec dens(bins);
    arma::vec mean_bin(bins);
    M1.fill(NA_REAL);
    eM1.fill(NA_REAL);
    M2.fill(NA_REAL);
    eM2.fill(NA_REAL);
    M4.fill(NA_REAL);
    D1.fill(NA_REAL);
    eD1.fill(NA_REAL);
    D2.fill(NA_REAL);
    eD2.fill(NA_REAL);
    D4.fill(NA_REAL);
    mean_bin.zeros();

#pragma omp parallel default(shared)
{
    #pragma omp for
    for (int i = 0; i < bins; i++) {
        arma::vec sum_m1(nsteps);
        arma::vec sum_m2(nsteps);
        arma::vec sum_m4(nsteps);
        arma::vec len_step(nsteps);
        sum_m1.zeros();
        sum_m2.zeros();
        sum_m4.zeros();
        len_step.zeros();
        double len_bin = 0;
        for (int n = 0; n < data.n_elem - steps.max(); n++) {
            if(data(n) >= U(i) && data(n) < U(i+1) && arma::is_finite(data(n))) {
                for (int s = 0; s < nsteps; s++) {
                    if(arma::is_finite(data(n+steps(s)))) {
                        double inc = data(n+steps(s)) - data(n);
                        sum_m1(s) += inc;
                        sum_m2(s) += inc*inc;
                        sum_m4(s) += inc*inc*inc*inc;
                        len_step(s)++;
                    }
                }
                mean_bin(i) += data(n);
                len_bin++;
            }
        }
        mean_bin(i) /= len_bin;
        dens(i) = arma::max(len_step);
        //
        if(len_bin >= bin_min) {
            M1.row(i) = arma::trans(sum_m1/len_step);
            M2.row(i) = arma::trans(sum_m2/len_step);
            M4.row(i) = arma::trans(sum_m4/len_step);
            // calculate the errors of M1 and M2
            eM1.row(i) = arma::trans(arma::sqrt((M2.row(i) - arma::square(M1.row(i)))/len_step));
            eM2.row(i) = arma::trans(arma::sqrt((M4.row(i) - arma::square(M2.row(i)))/len_step));

            // linear regression with weights to get D1, D2 and D4
            arma::vec coef = linreg(steps, arma::trans(M1.row(i)), 1/eM1.row(i));
            D1(i) = sf*coef(1);
            // 2 * tau * D2 = M2 - (tau * D1)^2
            arma::vec y = arma::trans(M2.row(i)) - arma::square(steps*coef(1));
            coef = linreg(steps, y, 1/eM2.row(i));
            D2(i) = sf*coef(1)/2;
            coef = linreg(steps, arma::trans(M4.row(i)));
            D4(i) = sf*coef(1)/24;
            // calculate the errors of D1 and D2
            eD1(i) = sqrt((2*sf*D2(i) - D1(i)*D1(i))/dens(i));
            eD2(i) = sqrt((2*sf*D4(i) - D2(i)*D2(i))/dens(i));
        }
    }
}
    List ret;
    ret["D1"] = D1;
    ret["eD1"] = eD1;
    ret["D2"] = D2;
    ret["eD2"] = eD2;
    ret["D4"] = D4;
    ret["mean_bin"] = mean_bin;
    ret["density"] = dens;
    ret["M1"] = M1;
    ret["eM1"] = eM1;
    ret["M2"] = M2;
    ret["eM2"] = eM2;
    ret["M4"] = M4;
    ret["U"] = U;
    return ret;
}
