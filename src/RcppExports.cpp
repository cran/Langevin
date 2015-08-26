// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// Langevin1D
List Langevin1D(const arma::vec& data, const int& bins, const arma::vec& steps, const double& sf, const int& bin_min, int reqThreads);
RcppExport SEXP Langevin_Langevin1D(SEXP dataSEXP, SEXP binsSEXP, SEXP stepsSEXP, SEXP sfSEXP, SEXP bin_minSEXP, SEXP reqThreadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const arma::vec& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const int& >::type bins(binsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type steps(stepsSEXP);
    Rcpp::traits::input_parameter< const double& >::type sf(sfSEXP);
    Rcpp::traits::input_parameter< const int& >::type bin_min(bin_minSEXP);
    Rcpp::traits::input_parameter< int >::type reqThreads(reqThreadsSEXP);
    __result = Rcpp::wrap(Langevin1D(data, bins, steps, sf, bin_min, reqThreads));
    return __result;
END_RCPP
}
// Langevin2D
List Langevin2D(const arma::mat& data, const int& bins, const arma::vec& steps, const double& sf, const int& bin_min, int reqThreads);
RcppExport SEXP Langevin_Langevin2D(SEXP dataSEXP, SEXP binsSEXP, SEXP stepsSEXP, SEXP sfSEXP, SEXP bin_minSEXP, SEXP reqThreadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const arma::mat& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const int& >::type bins(binsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type steps(stepsSEXP);
    Rcpp::traits::input_parameter< const double& >::type sf(sfSEXP);
    Rcpp::traits::input_parameter< const int& >::type bin_min(bin_minSEXP);
    Rcpp::traits::input_parameter< int >::type reqThreads(reqThreadsSEXP);
    __result = Rcpp::wrap(Langevin2D(data, bins, steps, sf, bin_min, reqThreads));
    return __result;
END_RCPP
}
// timeseries1D
NumericVector timeseries1D(const unsigned int& N, const double& startpoint, const double& d13, const double& d12, const double& d11, const double& d10, const double& d22, const double& d21, const double& d20, const double& sf, double dt);
RcppExport SEXP Langevin_timeseries1D(SEXP NSEXP, SEXP startpointSEXP, SEXP d13SEXP, SEXP d12SEXP, SEXP d11SEXP, SEXP d10SEXP, SEXP d22SEXP, SEXP d21SEXP, SEXP d20SEXP, SEXP sfSEXP, SEXP dtSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const unsigned int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< const double& >::type startpoint(startpointSEXP);
    Rcpp::traits::input_parameter< const double& >::type d13(d13SEXP);
    Rcpp::traits::input_parameter< const double& >::type d12(d12SEXP);
    Rcpp::traits::input_parameter< const double& >::type d11(d11SEXP);
    Rcpp::traits::input_parameter< const double& >::type d10(d10SEXP);
    Rcpp::traits::input_parameter< const double& >::type d22(d22SEXP);
    Rcpp::traits::input_parameter< const double& >::type d21(d21SEXP);
    Rcpp::traits::input_parameter< const double& >::type d20(d20SEXP);
    Rcpp::traits::input_parameter< const double& >::type sf(sfSEXP);
    Rcpp::traits::input_parameter< double >::type dt(dtSEXP);
    __result = Rcpp::wrap(timeseries1D(N, startpoint, d13, d12, d11, d10, d22, d21, d20, sf, dt));
    return __result;
END_RCPP
}
// timeseries2D
arma::mat timeseries2D(const unsigned int& N, const double& startpointx, const double& startpointy, const arma::mat& D1_1, const arma::mat& D1_2, const arma::mat& g_11, const arma::mat& g_12, const arma::mat& g_21, const arma::mat& g_22, const double& sf, double dt);
RcppExport SEXP Langevin_timeseries2D(SEXP NSEXP, SEXP startpointxSEXP, SEXP startpointySEXP, SEXP D1_1SEXP, SEXP D1_2SEXP, SEXP g_11SEXP, SEXP g_12SEXP, SEXP g_21SEXP, SEXP g_22SEXP, SEXP sfSEXP, SEXP dtSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const unsigned int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< const double& >::type startpointx(startpointxSEXP);
    Rcpp::traits::input_parameter< const double& >::type startpointy(startpointySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type D1_1(D1_1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type D1_2(D1_2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type g_11(g_11SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type g_12(g_12SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type g_21(g_21SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type g_22(g_22SEXP);
    Rcpp::traits::input_parameter< const double& >::type sf(sfSEXP);
    Rcpp::traits::input_parameter< double >::type dt(dtSEXP);
    __result = Rcpp::wrap(timeseries2D(N, startpointx, startpointy, D1_1, D1_2, g_11, g_12, g_21, g_22, sf, dt));
    return __result;
END_RCPP
}