#ifndef INIT
#define INIT

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"
#include <EigenRand/EigenRand>

#include "eigenExtension.h"

void simple(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed='r');
void uniform(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, double const& a, double const& b, unsigned const seed='r');
void normal(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, double const& mu, double const& sigma, unsigned const seed='r');

void Xavier(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed='r');
void He(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed='r');
void Kaiming(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed='r');
void Bergio(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed='r');


void initialisation(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<double> const& supParameters,
std::string const& generator, unsigned const seed='r');

#endif
