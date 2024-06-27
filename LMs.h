#ifndef LMS
#define LMS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "propagation.h"
#include "scaling.h"
#include "utilities.h"
#include "eigenExtension.h"

std::map<std::string,Sdouble> LM_base(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& mu, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LM(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble& mu, Sdouble& factor, Sdouble const& eps, int const& maxIter, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMBall(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble& mu, Sdouble& factor, Sdouble const& eps, int const& maxIter, std::string const& norm, Sdouble const& radiusBall, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMF(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& eps, int const& maxIter, Sdouble const& RMin, Sdouble const& RMax, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMMore(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& eps, int const& maxIter, Sdouble const& sigma, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMNielson(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& eps, int const& maxIter, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& epsDiag,
bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMUphill(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& eps, int const& maxIter, Sdouble const& RMin, Sdouble const& RMax, int const& b, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LMPerso(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& eps, int const& maxIter, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& epsDiag, bool const record=false, std::string const fileExtension="");


//-- --------------------------------------------------------- Initialisation simple -------------------------------------------------------------------------------------------------------------

std::map<std::string,Sdouble> init(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons,std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
bool const record=false, std::string const fileExtension="");

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

std::map<std::string,Sdouble> train_LM(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");


#endif
