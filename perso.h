#ifndef PERSO
#define PERSO

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <iomanip>

#include <random>
#include <algorithm>
#include "unistd.h"

#include <Eigen/Dense>

#include "propagation.h"
#include "utilities.h"

#include "classic.h"


//---------------------------------------------- Inegalité sur la vitesse de dissipation-------------------------------------------------------------
std::map<std::string,double> LC_EGD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LCI_EGD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string, double> LC_EGD2(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::VectorXd> &bias, std::string const &type_perte, double const &learning_rate_init,
double const &eps, int const &maxEpoch,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, double> LC_Mechanic(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, 
std::vector<int> const &globalIndices,std::vector<std::string> const &activations, std::vector<Eigen::MatrixXd> &weights, 
std::vector<Eigen::VectorXd> &bias, std::string const &type_perte, double const &learning_rate_init, double const &eps, int const &maxEpoch,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, double> LC_M(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::VectorXd> &bias, std::string const &type_perte, double const &learning_rate_init,
double const &eps, int const &maxEpoch,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, double> LC_RMSProp(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::VectorXd> &bias, std::string const &type_perte, double const &learning_rate_init,
double const &eps, int const &maxEpoch,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, double> LC_clipping(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::VectorXd> &bias, std::string const &type_perte, double const &learning_rate_init,
double const &eps, int const &maxEpoch,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, double> LC_signGD(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::VectorXd> &bias, std::string const &type_perte, double const &learning_rate_init,
double const &eps, int const &maxEpoch,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, double> LC_EGD3(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::VectorXd> &bias, std::string const &type_perte, double const &learning_rate_init,
double const &eps, int const &maxEpoch,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

//----------------------------------------------- Egalité vitesse de dissipation--------------------------------------------------------------------------------

std::map<std::string,double> LC_EM(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, 
double const& learning_rate_init, double const& beta1_init, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension);

//---------------------------------------------- Approche decroissance ----------------------------------------------------

std::map<std::string,double> GD_Em(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> Momentum_Em(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& beta1_init, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

//------------------------------------------- Schéma adaptatif classique ----------------------------------------------------------------------

std::map<std::string,double> EulerRichardson(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> train_Perso(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, double const& beta1, double const& beta2, int const& batch_size, double const& mu_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");


#endif
