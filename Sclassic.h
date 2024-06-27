#ifndef SCLASSIC
#define SCLASSIC

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

#include "propagation.h"
#include "eigenExtension.h"

std::map<std::string,Sdouble> SGD(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> SGD_AllBatches(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> SAdam(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> SAdam_AllBatches(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> SAdam_WB(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> train_Sclassic(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate, int const& batch_size, unsigned const Sseed, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension);

#endif
