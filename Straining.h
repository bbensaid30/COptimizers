#ifndef STRAINING
#define STRAINING

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <random>
#include <algorithm>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "Sclassic.h"
#include "Sperso.h"

std::map<std::string,Sdouble> Strain(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo,Sdouble const& eps, int const& maxIter, Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2,
int const& batch_size, unsigned const Sseed,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

#endif
