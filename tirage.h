#ifndef TIRAGE
#define TIRAGE

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include <omp.h>

#include "init.h"
#include "propagation.h"
#include "training.h"


std::vector<std::map<std::string,double>> tiragesRegression(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2, int const & batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
bool const tracking=false);

std::vector<std::map<std::string,double>> tiragesClassification(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2, int const & batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
bool const tracking=false);

void minsRecordRegression(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps);
void minsRecordClassification(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps);

void predictionsRecord(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2, int const & batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
std::string const& folder, std::string const fileExtension="", bool const tracking=false, bool const track_continuous=false);


std::string informationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eta, int const& batch_size, double const& eps, int const& maxEpoch, std::string const fileExtension="");

void classificationRate(Eigen::MatrixXd const& dataTrain, Eigen::MatrixXd const& dataTest, Eigen::MatrixXd const& AsTrain, Eigen::MatrixXd const& AsTest,
int const& PTrain, int const& PTest, double& rateTrain, double& rateTest);

#endif
