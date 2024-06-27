#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <numeric>

#include <Eigen/Dense>

#include <omp.h>

double sum_tab(double *tab, int const& taille);
double sum_decale(double *tab, int const& taille, int i);
int cyclic(int i, int m);
int indice_max_tab(double *tab, int const& taille);
double mean_tab(double *tab, int const& taille);
double sum_max_tab(double *tab, int const& taille, double& max);
void affiche_tab(double *tab, int const& taille);
void init_grad_zero(std::vector<Eigen::VectorXd>& grads, int m, int N);
int selection_data(int const& i, int const& m, int const& batch_size, int const& P, Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, Eigen::MatrixXd& echantillonX, Eigen::MatrixXd& echantillonY);

double max(double a, double b);

double dInf(Eigen::MatrixXd const& A, Eigen::MatrixXd const& B);
double MAE(Eigen::MatrixXd const& A, Eigen::MatrixXd const& B);
double MAPE(Eigen::MatrixXd const& A, Eigen::MatrixXd const& B);

void echanger(double& a, double& b);
bool appartient_intervalle(double x, double gauche, double droite);

int proportion(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, std::vector<double>& proportions,  std::vector<double>& distances, double const& epsNeight);
int numero_point(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, double const& epsNeight);

double mean(std::vector<int> const& values);
double sd(std::vector<int> const& values, double const& moy);
double median(std::vector<double>& values);
int median(std::vector<int>& values);
double minVector(std::vector<double> const& values);
int minVector(std::vector<int> const& values);

double norme(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::string const norm="2");
double distance(std::vector<Eigen::MatrixXd> const& weightsPrec, std::vector<Eigen::VectorXd> const& biasPrec,
std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::string const norm="2");

double cosVector(Eigen::VectorXd const& v1, Eigen::VectorXd const& v2);

void convexCombination(std::vector<Eigen::MatrixXd>& weightsMoy, std::vector<Eigen::VectorXd>& biasMoy, std::vector<Eigen::MatrixXd> const& weights,
std::vector<Eigen::VectorXd> const& bias, int const& L, double const& lambda);
void RKCombination(std::vector<Eigen::MatrixXd>& weightsInter, std::vector<Eigen::VectorXd>& biasInter, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, int const& L);

void nesterovCombination(std::vector<Eigen::MatrixXd> const& weights1, std::vector<Eigen::VectorXd> const& bias1, std::vector<Eigen::MatrixXd> const& weights2,
std::vector<Eigen::VectorXd> const& bias2, std::vector<Eigen::MatrixXd>& weightsInter, std::vector<Eigen::VectorXd>& biasInter, int const& L, double const& lambda);

void tabToVector(std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::VectorXd& point);

void standardization(Eigen::MatrixXd& X);

int nbLines(std::ifstream& flux);
void readMatrix(std::ifstream& flux, Eigen::MatrixXd& result, int const& nbRows, int const& nbCols);
void readVector(std::ifstream& flux, Eigen::VectorXd& result, int const& nbRows);

double indexProperValues(Eigen::MatrixXd const& H);

double expInv(double const& x);

double fAdam(double const& a, double const& b, double const& t);


#endif
