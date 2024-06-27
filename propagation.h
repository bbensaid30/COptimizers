#ifndef PROPAGATION
#define PROPAGATION

#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include "activations.h"
#include "perte.h"


//------------------------------------------------------------------ Propagation directe ----------------------------------------------------------------------------------------

void fforward(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes);

double risk(Eigen::MatrixXd const& Y, int const& P, Eigen::MatrixXd const& output_network, std::string const& type_perte, bool const normalized=true);

//-------------------------------------------------------------------- Rétropropagation -----------------------------------------------------------------------------------------------

void backward(Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::VectorXd& gradient, std::string const& type_perte, bool const normalized=true);

void QSO_backward(Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes,Eigen::VectorXd& gradient, Eigen::MatrixXd& Q, std::string const& type_perte);

void QSO_backwardJacob(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& J);


//-------------------------------------------------------------- Mise à jour ---------------------------------------------------------------------------------------------

void solve(Eigen::VectorXd const& gradient, Eigen::MatrixXd const& hessian, Eigen::VectorXd& delta, std::string const method = "HouseholderQR");

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
Eigen::VectorXd const& delta);

void updateNesterov(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& weights2, std::vector<Eigen::VectorXd>& bias2, Eigen::VectorXd const& delta, double const& lambda1, double const& lambda2);

#endif
