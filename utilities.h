#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <numeric>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include <omp.h>

#include "eigenExtension.h"

Sdouble dInf(Eigen::SMatrixXd const& A, Eigen::SMatrixXd const& B);
Sdouble MAE(Eigen::SMatrixXd const& A, Eigen::SMatrixXd const& B);
Sdouble MAPE(Eigen::SMatrixXd const& A, Eigen::SMatrixXd const& B);

void echanger(Sdouble& a, Sdouble& b);
bool appartient_intervalle(Sdouble x, Sdouble gauche, Sdouble droite);

int proportion(Eigen::SVectorXd const& currentPoint, std::vector<Eigen::SVectorXd> const& points, std::vector<Sdouble>& proportions,  std::vector<Sdouble>& distances, Sdouble const& epsNeight);
int numero_point(Eigen::SVectorXd const& currentPoint, std::vector<Eigen::SVectorXd> const& points, Sdouble const& epsNeight);

Sdouble mean(std::vector<Sdouble> const& values);
Sdouble mean(std::vector<int> const& values);
Sdouble sd(std::vector<Sdouble> const& values, Sdouble const& moy);
Sdouble sd(std::vector<int> const& values, Sdouble const& moy);
Sdouble median(std::vector<Sdouble>& values);
int median(std::vector<int>& values);
Sdouble minVector(std::vector<Sdouble> const& values);
int minVector(std::vector<int> const& values);

Sdouble norme(std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::string const norm="2");
Sdouble distance(std::vector<Eigen::SMatrixXd> const& weightsPrec, std::vector<Eigen::SVectorXd> const& biasPrec,
std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, std::string const norm="2");

Sdouble cosVector(Eigen::SVectorXd const& v1, Eigen::SVectorXd const& v2);

void convexCombination(std::vector<Eigen::SMatrixXd>& weightsMoy, std::vector<Eigen::SVectorXd>& biasMoy, std::vector<Eigen::SMatrixXd> const& weights,
std::vector<Eigen::SVectorXd> const& bias, int const& L, Sdouble const& lambda);
void RKCombination(std::vector<Eigen::SMatrixXd>& weightsInter, std::vector<Eigen::SVectorXd>& biasInter, std::vector<Eigen::SMatrixXd> const& weights, std::vector<Eigen::SVectorXd> const& bias, int const& L);

void nesterovCombination(std::vector<Eigen::SMatrixXd> const& weights1, std::vector<Eigen::SVectorXd> const& bias1, std::vector<Eigen::SMatrixXd> const& weights2,
std::vector<Eigen::SVectorXd> const& bias2, std::vector<Eigen::SMatrixXd>& weightsInter, std::vector<Eigen::SVectorXd>& biasInter, int const& L, Sdouble const& lambda);

void tabToVector(std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::SVectorXd& point);

void standardization(Eigen::SMatrixXd& X);

int nbLines(std::ifstream& flux);
void readMatrix(std::ifstream& flux, Eigen::SMatrixXd& result, int const& nbRows, int const& nbCols);
void readVector(std::ifstream& flux, Eigen::SVectorXd& result, int const& nbRows);

Sdouble indexProperValues(Eigen::SMatrixXd const& H);

Sdouble expInv(Sdouble const& x);

Sdouble fAdam(Sdouble const& a, Sdouble const& b, Sdouble const& t);



#endif
