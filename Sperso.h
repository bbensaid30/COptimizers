#ifndef SPERSO
#define SPERSO

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>

#include <random>
#include <algorithm>

#include <Eigen/Dense>

#include "propagation.h"
#include "utilities.h"
#include "Sclassic.h"

std::map<std::string,double> SLCEGD(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> SLCEGDR(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> train_Sperso(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension);

#endif