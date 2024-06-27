#ifndef STIRAGE
#define STIRAGE

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

#include <omp.h>

#include "init.h"
#include "propagation.h"
#include "tirage.h"
#include "Straining.h"
#include "eigenExtension.h"


std::vector<std::map<std::string,Sdouble>> StiragesRegression(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking=false);

std::vector<std::map<std::string,Sdouble>> StiragesClassification(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking=false);

void SminsRecordRegression(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps);
void SminsRecordClassification(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps);

std::string SinformationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eta, int const& batch_size, Sdouble const& eps, int const& maxIter, std::string const fileExtension="");

#endif