#ifndef STIRAGE
#define STIRAGE

#include <iostream>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include <omp.h>

#include "init.h"
#include "propagation.h"
#include "tirage.h"
#include "Straining.h"


std::vector<std::map<std::string,double>> StiragesRegression(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking=false);

std::vector<std::map<std::string,double>> StiragesClassification(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking=false);

void SminsRecordRegression(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps);
void SminsRecordClassification(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps);

std::string SinformationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eta, int const& batch_size, double const& eps, int const& maxEpoch, std::string const fileExtension="");

#endif