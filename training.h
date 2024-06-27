#ifndef TRAINING
#define TRAINING

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <random>
#include <algorithm>

#include <Eigen/Dense>

#include "classic.h"
#include "LMs.h"
#include "perso.h"
#include "incremental.h"
#include "essai.h"

std::map<std::string,double> train(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo,double const& eps, int const& maxEpoch, double const& learning_rate, 
double const& clip,double const& seuil, double const& beta1, double const& beta2,int const& batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

#endif
