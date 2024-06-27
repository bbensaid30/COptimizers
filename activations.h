#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <string>
#include <cmath>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

//Activations classiques
void linear(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void sigmoid(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void softmax(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, int const q=-1);
void tanh(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void reLU(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void GELU(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void softplus(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void IDC(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void sinus(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);

//Activations pour les cas tests de la fonction de cout norme 2
//L'hypothèse est vérifiée pour tous les mins pour polyTwo et pour le min global pour polyThree
void polyTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c=0);
void polyThree(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c=0);
void polyFour(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c=0);
void polyFive(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);
void polyEight(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);

void expTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c=0);
void expFour(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c=0);
void expFive(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c=0);

//Activations pour les cas tests de la fonction de cout entropie
void ratTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c=0);
void cloche(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S);

void activation(std::string nameActivation, Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, int const q=-1);

#endif

