#ifndef PERSO
#define PERSO

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <random>
#include <algorithm>
#include "unistd.h"

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include "propagation.h"
#include "eigenExtension.h"
#include "utilities.h"

#include "classic.h"

std::map<std::string,Sdouble> splitting_LCEGD(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, int const& batch_size, Sdouble const& f1, Sdouble const& f2, Sdouble const& eps, int const& maxIter,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> splitting2_LCEGD(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, int const& batch_size, Sdouble const& f1, Sdouble const& f2, Sdouble const& eps, int const& maxIter,
bool const tracking =false, bool const record=false, std::string const fileExtension="");

//---------------------------------------------- Inegalité sur la vitesse de dissipation-------------------------------------------------------------
std::map<std::string,Sdouble> LC_EGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> LCI_EGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string, Sdouble> LC_EGD2(Eigen::SMatrixXd &X, Eigen::SMatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::SMatrixXd> &weights, std::vector<Eigen::SVectorXd> &bias, std::string const &type_perte, Sdouble const &learning_rate_init,
Sdouble const &eps, int const &maxIter,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, Sdouble> LC_Mechanic(Eigen::SMatrixXd &X, Eigen::SMatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, 
std::vector<int> const &globalIndices,std::vector<std::string> const &activations, std::vector<Eigen::SMatrixXd> &weights, 
std::vector<Eigen::SVectorXd> &bias, std::string const &type_perte, Sdouble const &learning_rate_init, Sdouble const &eps, int const &maxIter,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, Sdouble> LC_M(Eigen::SMatrixXd &X, Eigen::SMatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::SMatrixXd> &weights, std::vector<Eigen::SVectorXd> &bias, std::string const &type_perte, Sdouble const &learning_rate_init,
Sdouble const &eps, int const &maxIter,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, Sdouble> LC_clipping(Eigen::SMatrixXd &X, Eigen::SMatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::SMatrixXd> &weights, std::vector<Eigen::SVectorXd> &bias, std::string const &type_perte, Sdouble const &learning_rate_init,
Sdouble const &eps, int const &maxIter,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, Sdouble> LC_signGD(Eigen::SMatrixXd &X, Eigen::SMatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::SMatrixXd> &weights, std::vector<Eigen::SVectorXd> &bias, std::string const &type_perte, Sdouble const &learning_rate_init,
Sdouble const &eps, int const &maxIter,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

std::map<std::string, Sdouble> LC_EGD3(Eigen::SMatrixXd &X, Eigen::SMatrixXd &Y, int const &L, std::vector<int> const &nbNeurons, std::vector<int> const &globalIndices,
std::vector<std::string> const &activations, std::vector<Eigen::SMatrixXd> &weights, std::vector<Eigen::SVectorXd> &bias, std::string const &type_perte, Sdouble const &learning_rate_init,
Sdouble const &eps, int const &maxIter,
bool const tracking = false, bool const record = false, std::string const fileExtension = "");

//----------------------------------------------- Egalité vitesse de dissipation--------------------------------------------------------------------------------

std::map<std::string,Sdouble> LC_EM(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, 
Sdouble const& learning_rate_init, Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension);

//---------------------------------------------- Approche decroissance ----------------------------------------------------

std::map<std::string,Sdouble> GD_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> Momentum_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");

//------------------------------------------- Schéma adaptatif classique ----------------------------------------------------------------------

std::map<std::string,Sdouble> EulerRichardson(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const record=false, std::string const fileExtension="");

std::map<std::string,Sdouble> train_Perso(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate_init, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, Sdouble const& mu_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking=false, bool const track_continuous=false, bool const record=false, std::string const fileExtension="");


#endif
