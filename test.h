#ifndef TEST
#define TEST

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include "init.h"
#include "training.h"
#include "utilities.h"


void test_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor,
double const& Rlim, double const& RMin, double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap,
double const& alpha, double const& pas, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor,
double const& Rlim, double const& RMin, double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap,
double const& alpha, double const& pas, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_PolyFour(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor,
double const& Rlim, double const& RMin, double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap,
double const& alpha, double const& pas, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_PolyFive(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor,
double const& Rlim, double const& RMin, double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap,
double const& alpha, double const& pas, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const setHyperparameters="");

void test_PolyEight(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor,
double const& Rlim, double const& RMin, double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap,
double const& alpha, double const& pas, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");



void test_Cloche(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor,
double const& Rlim, double const& RMin, double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap,
double const& alpha, double const& pas, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_RatTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor,
double const& Rlim, double const& RMin, double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap,
double const& alpha, double const& pas, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

#endif
