#ifndef LMS
#define LMS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>

#include <Eigen/Dense>

#include "propagation.h"
#include "scaling.h"
#include "utilities.h"

std::map<std::string,double> LM_base(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& mu, double const& eps, int const& maxEpoch,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LM(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double& mu, double& factor, double const& eps, int const& maxEpoch, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMBall(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double& mu, double& factor, double const& eps, int const& maxEpoch, std::string const& norm, double const& radiusBall, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMF(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& eps, int const& maxEpoch, double const& RMin, double const& RMax, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMMore(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& eps, int const& maxEpoch, double const& sigma, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMNielson(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& eps, int const& maxEpoch, double const& tau, double const& beta, double const& gamma, int const& p, double const& epsDiag,
bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMUphill(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& eps, int const& maxEpoch, double const& RMin, double const& RMax, int const& b, bool const record=false, std::string const fileExtension="");

std::map<std::string,double> LMPerso(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& eps, int const& maxEpoch, double const& RMin, double const& RMax, int const& b, double const& epsDiag, bool const record=false, std::string const fileExtension="");


//-- --------------------------------------------------------- Initialisation simple -------------------------------------------------------------------------------------------------------------

std::map<std::string,double> init(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons,std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
bool const record=false, std::string const fileExtension="");

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

std::map<std::string,double> train_LM(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& eps, int const& maxEpoch, double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
double const& tau, double const& beta, double const& gamma, int const& p,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");


#endif
