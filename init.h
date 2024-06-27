#ifndef INIT
#define INIT

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>


void simple(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, unsigned const seed='r');
void uniform(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const& a, double const& b, unsigned const seed='r');
void normal(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, double const& mu, double const& sigma, unsigned const seed='r');

void Xavier(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, unsigned const seed='r');
void He(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, unsigned const seed='r');
void Kaiming(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, unsigned const seed='r');
void Bergio(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, unsigned const seed='r');


void initialisation(std::vector<int> const& nbNeurons, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<double> const& supParameters,
std::string const& generator, unsigned const seed='r');

#endif
