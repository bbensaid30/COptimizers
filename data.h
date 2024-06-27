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
#include <EigenRand/EigenRand>

std::vector<Eigen::MatrixXd> sineWave(int const& nbPoints);
std::vector<Eigen::MatrixXd> squareWave(int const& nbPoints, double const frequence=1);
std::vector<Eigen::MatrixXd> sinc1(int const& nbPoints);
std::vector<Eigen::MatrixXd> square(int const& nbPoints);
std::vector<Eigen::MatrixXd> squareRoot(int const& nbPoints);
std::vector<Eigen::MatrixXd> exp(int const& nbPoints);


std::vector<Eigen::MatrixXd> sinc2(int const& nbPoints);
std::vector<Eigen::MatrixXd> exp2(int const& nbPoints);
std::vector<Eigen::MatrixXd> carreFunction2(int const& nbPoints);

std::vector<Eigen::MatrixXd> twoSpiral(int const& nbPoints);
std::vector<Eigen::MatrixXd> twoSpiralOriginal();
std::vector<Eigen::MatrixXd> Boston(int const PTrain=404, int const PTest=102);
std::vector<Eigen::MatrixXd> Sonar(int const PTrain=104, int const PTest=104);
std::vector<Eigen::MatrixXd> German(int const PTrain=900, int const PTest=100);
std::vector<Eigen::MatrixXd> Diamonds(int const PTrain=37760, int const PTest=16183);
std::vector<Eigen::MatrixXd> California(int const PTrain=11272, int const PTest=2818);
std::vector<Eigen::MatrixXd> MNIST(int const PTrain=60000, int const PTest=10000);


std::vector<Eigen::MatrixXd> trainTestData(std::vector<Eigen::MatrixXd> const& data, double const& percTrain = 0.9, bool const reproductible = true);

#endif
