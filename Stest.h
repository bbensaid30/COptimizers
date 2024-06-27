#ifndef STEST
#define STEST

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include "init.h"
#include "Straining.h"
#include "utilities.h"

void Stest_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& beta1, double const& beta2,
int const& batch_size, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const record=false, std::string const setHyperparameters="");

void Stest_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& beta1, double const& beta2,
int const& batch_size, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const record=false, std::string const setHyperparameters="");

void Stest_PolyFive(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& beta1, double const& beta2,
int const& batch_size, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const record=false, std::string const setHyperparameters="");


#endif