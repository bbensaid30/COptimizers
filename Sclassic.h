#ifndef SCLASSIC
#define SCLASSIC

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

std::map<std::string,double> SGD(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> SGD_AllBatches(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> SAdam(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> SAdam_AllBatches(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> SAdam_WB(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> SAdam_AllBatches_WB(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension);

std::map<std::string,double> train_Sclassic(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate, int const& batch_size, unsigned const Sseed, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension);

#endif
