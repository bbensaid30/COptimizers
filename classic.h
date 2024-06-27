#ifndef CLASSIC
#define CLASSIC

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

std::map<std::string,double> GD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, int const& batch_size, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> Momentum(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta1, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");


std::map<std::string,double> RMSProp(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> Adam_WB(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> Adam(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> train_classic(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate, double const& clip, int const& batch_size, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");



#endif
