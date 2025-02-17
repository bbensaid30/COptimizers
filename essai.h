#ifndef ESSAI
#define ESSAI

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>

#include <random>
#include <algorithm>
#include "unistd.h"

#include <Eigen/Dense>

#include "propagation.h"
#include "classic.h"
#include "utilities.h"

std::map<std::string,double> HolderHB(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension);

std::map<std::string,double> LC_RK(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> RK(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> PGD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> PGD_Brent(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> PGD0(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> PGRK2(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> PM(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& beta1_init, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> PER(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> ER_Em(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> ERIto(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LM_ER(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& mu_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> Momentum_Verlet(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
double const& beta1, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> RK2Momentum(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
double const& beta1, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> Momentum_Em_parametric(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& beta1_init, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> ABE(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& beta1_init, double const& beta2_init, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension);

std::map<std::string,double> Nesterov2(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
double const& eps, int const& maxEpoch, bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> train_Essai(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, double const& beta1, double const& beta2, double const& mu_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

#endif