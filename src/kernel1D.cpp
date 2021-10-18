// Copyright (C) 2021  Philip Rinn
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

using namespace Rcpp;

// Epanechnikov kernel
arma::vec kappa(const arma::vec& data) {
    arma::vec res = arma::zeros<arma::vec>(data.n_elem);
    arma::uvec hit = arma::find(data%data <= 5);
    res(hit) = 0.75 * (1 - data(hit)%data(hit)/5) / std::sqrt(5);
    return res;
}

// Nadaraya-Watson estimator
// [[Rcpp::export(".kernel1D")]]
List kernel1D(const arma::vec& data, const int& bins, const double& sf, const double& h) {
    arma::vec ests = arma::linspace<arma::vec>(data.min(), data.max(), bins);
    arma::vec ka_sum(bins);
    arma::vec wa_sum(bins);
    arma::vec wa2_sum(bins);
    arma::vec wa4_sum(bins);
    arma::vec phi(bins);
    ka_sum.fill(NA_REAL);
    wa_sum.fill(NA_REAL);
    wa2_sum.fill(NA_REAL);
    wa4_sum.fill(NA_REAL);
    phi.fill(NA_REAL);

    arma::vec data_min = data.head(data.n_elem - 1);
    arma::vec data_inc = data.tail(data.n_elem - 1) - data_min;

    for(int i = 0; i < bins; i++) {
        arma::vec data_kappa = kappa((ests(i) - data_min)/h)/h;
        ka_sum(i) = arma::sum(data_kappa);
        wa_sum(i) = arma::sum(data_inc%data_kappa);
        wa2_sum(i) = arma::sum(data_inc%data_inc%data_kappa);
        wa4_sum(i) = arma::sum(data_inc%data_inc%data_inc%data_inc%data_kappa);
    }

    arma::vec M1 = wa_sum / ka_sum;
    arma::vec M2 = wa2_sum / ka_sum;
    arma::vec M4 = wa4_sum / ka_sum;
    arma::vec eD1 = sf * arma::sqrt((M2 - arma::square(M1)) / ka_sum);
    arma::vec eD2 = (sf/2) * arma::sqrt((M4 - arma::square(M2)) / ka_sum);
    arma::vec D1 = sf * M1;
    arma::vec D2 = (sf/2) * M2;
    arma::vec D4 = (sf/24) * M4;

    List ret;
    ret["D1"] = D1;
    ret["eD1"] = eD1;
    ret["D2"] = D2;
    ret["eD2"] = eD2;
    ret["D4"] = D4;
    ret["mean_bin"] = ests;

    ret.attr("class") = CharacterVector::create("Langevin");
    return ret;
}
