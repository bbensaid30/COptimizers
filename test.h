#ifndef TEST
#define TEST

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "init.h"
#include "training.h"
#include "utilities.h"


void test_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_PolyFour(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_PolyFive(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const setHyperparameters="");

void test_PolyEight(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");



void test_Cloche(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

void test_RatTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo,
std::string const& algo, Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble& mu, Sdouble& factor,
Sdouble const& Rlim, Sdouble const& RMin, Sdouble const& RMax, Sdouble const& epsDiag, int const& b, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap,
Sdouble const& alpha, Sdouble const& pas, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const tracking=false, bool const track_continuous=false, bool const record=false,std::string const setHyperparameters="");

#endif
