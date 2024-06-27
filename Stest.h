#ifndef STEST
#define STEST

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "init.h"
#include "Straining.h"
#include "utilities.h"

void Stest_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const record=false, std::string const setHyperparameters="");

void Stest_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const record=false, std::string const setHyperparameters="");

void Stest_PolyFive(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, Sdouble const& eps, int const& maxIter, Sdouble const& epsNeight,
bool const record=false, std::string const setHyperparameters="");


#endif