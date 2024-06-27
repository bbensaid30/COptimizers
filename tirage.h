#ifndef TIRAGE
#define TIRAGE

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include <omp.h>

#include "init.h"
#include "propagation.h"
#include "training.h"
#include "eigenExtension.h"


std::vector<std::map<std::string,Sdouble>> tiragesRegression(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
bool const tracking=false);

std::vector<std::map<std::string,Sdouble>> tiragesClassification(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
bool const tracking=false);

void minsRecordRegression(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps);
void minsRecordClassification(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps);

void predictionsRecord(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
std::string const& folder, std::string const fileExtension="", bool const tracking=false, bool const track_continuous=false);


std::string informationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eta, Sdouble const& eps, int const& maxIter, std::string const fileExtension="");

void classificationRate(Eigen::SMatrixXd const& dataTrain, Eigen::SMatrixXd const& dataTest, Eigen::SMatrixXd const& AsTrain, Eigen::SMatrixXd const& AsTest,
int const& PTrain, int const& PTest, Sdouble& rateTrain, Sdouble& rateTest);

#endif
