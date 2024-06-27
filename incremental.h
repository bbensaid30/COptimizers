#ifndef INCREMENTAL
#define INCREMENTAL

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <iterator>

#include <random>
#include <algorithm>
#include "unistd.h"

#include <Eigen/Dense>

#include "propagation.h"
#include "utilities.h"

std::map<std::string,double> RAG(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> RAG_L(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> RAG_L_ancien(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");


std::map<std::string,double> train_Incremental(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, double const& beta1, double const& beta2, int const& batch_size, double const& mu_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");


#endif