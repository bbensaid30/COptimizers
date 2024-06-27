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

#include "Sclassic.h"
#include "Sperso.h"

std::map<std::string,double> Strain(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo,double const& eps, int const& maxEpoch, double const& learning_rate, double const& beta1, double const& beta2,
int const& batch_size, unsigned const Sseed,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

#endif
