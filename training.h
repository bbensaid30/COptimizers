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
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "classic.h"
#include "LMs.h"
#include "perso.h"
#include "essai.h"

std::map<std::string,Sdouble> train(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo,Sdouble const& eps, int const& maxIter, Sdouble const& learning_rate, 
Sdouble const& clip,Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2,int const& batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

#endif
