// Copyright (C) 2012 - 2015  Philip Rinn
// Copyright (C) 2012  David Bastine
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

#include <Rcpp.h>
using namespace Rcpp;

//' Generate a 1D Langevin process
//'
//' \code{timeseries1D} generates a one-dimensional Langevin process using a
//' simple Euler integration. The drift function is a cubic polynomial, the
//' diffusion funcation a quadratic.
//'
//'
//' @param N a scalar denoting the length of the time series to generate.
//' @param startpoint a scalar denoting the starting point of the time series.
//' @param d13,d12,d11,d10 scalars denoting the coefficients for the drift polynomial.
//' @param d22,d21,d20 scalars denoting the coefficients for the diffusion polynomial.
//' @param sf a scalar denoting the sampling frequency.
//' @param dt a scalar denoting the maximal time step of integration. Default
//' \code{dt=0} yields \code{dt=1/sf}.
//'
//' @return \code{timeseries1D} returns a vector of length \code{N} with the
//' generated time series.
//'
//' @author Philip Rinn
//' @seealso \code{\link{timeseries2D}}
//' @examples
//' # Generate standardized Ornstein-Uhlenbeck-Process (d11=-1, d20=1)
//' # with integration time step 0.01 and sampling frequency 1
//' s <- timeseries1D(N=1e4, sf=1, dt=0.01);
//' t <- 1:1e4;
//' plot(t, s, t="l", main=paste("mean:", mean(s), " var:", var(s)));
//' @import Rcpp
//' @useDynLib Langevin
//' @export
// [[Rcpp::export]]
NumericVector timeseries1D(const unsigned int& N, const double& startpoint=0,
                           const double& d13=0, const double& d12=0,
                           const double& d11=-1, const double& d10=0,
                           const double& d22=0, const double& d21=0,
                           const double& d20=1, const double& sf=1000,
                           double dt=0) {
    NumericVector ts(N, NA_REAL);
    ts[0] = startpoint;

    // Calculate the integration time step and related values
    double stime = 1.0/sf;
    if(stime < dt || dt == 0) {
        dt = stime;
    }
    // Ration between sampling time and integration time step
    unsigned int m = ceil(stime/dt);
    dt = stime/m;

    // Integration
    double gamma;
    double x = ts[0];

    for (unsigned int i = 0; i < N; i++)  {
        // Integrate m steps and just save the last
        for (unsigned int j = 0; j < m; j++) {
            // Get a single Gaussian random number
            gamma = rnorm(1,0,std::sqrt(2))[0];
            // Iterate with integration time step dt
            x += (d13*std::pow(x,3) + d12*std::pow(x,2) + d11*x + d10)*dt + std::sqrt((d22*std::pow(x,2) + d21*x + d20)*dt)*gamma;
        }
        // Save every mth step
        ts[i] = x;
    }
    return ts;
}
