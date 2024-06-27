#ifndef DATA
#define DATA

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <ctime>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"
#include <EigenRand/EigenRand>

std::vector<Eigen::SMatrixXd> sineWave(int const& nbPoints);
std::vector<Eigen::SMatrixXd> squareWave(int const& nbPoints, Sdouble const frequence=1);
std::vector<Eigen::SMatrixXd> sinc1(int const& nbPoints);
std::vector<Eigen::SMatrixXd> square(int const& nbPoints);
std::vector<Eigen::SMatrixXd> squareRoot(int const& nbPoints);
std::vector<Eigen::SMatrixXd> exp(int const& nbPoints);


std::vector<Eigen::SMatrixXd> sinc2(int const& nbPoints);
std::vector<Eigen::SMatrixXd> exp2(int const& nbPoints);
std::vector<Eigen::SMatrixXd> carreFunction2(int const& nbPoints);

std::vector<Eigen::SMatrixXd> twoSpiral(int const& nbPoints);
std::vector<Eigen::SMatrixXd> twoSpiralOriginal();
std::vector<Eigen::SMatrixXd> Boston(int const PTrain=404, int const PTest=102);
std::vector<Eigen::SMatrixXd> Sonar(int const PTrain=104, int const PTest=104);
std::vector<Eigen::SMatrixXd> German(int const PTrain=900, int const PTest=100);
std::vector<Eigen::SMatrixXd> Diamonds(int const PTrain=37760, int const PTest=16183);
std::vector<Eigen::SMatrixXd> California(int const PTrain=11272, int const PTest=2818);
std::vector<Eigen::SMatrixXd> MNIST(int const PTrain=60000, int const PTest=10000);


std::vector<Eigen::SMatrixXd> trainTestData(std::vector<Eigen::SMatrixXd> const& data, Sdouble const& percTrain = 0.9, bool const reproductible = true);

#endif
