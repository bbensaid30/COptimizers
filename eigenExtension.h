#ifndef EIGEN_EXTENSION
#define EIGEN_EXTENSION

#include <cmath>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"


Eigen::SMatrixXd convertToShaman(Eigen::MatrixXd const& Md);
Eigen::MatrixXd convertToDouble(Eigen::SMatrixXd const& Md);

Sdouble accumul(std::vector<Sdouble> const& values);
Sdouble InnerProduct(std::vector<Sdouble> const& values1, std::vector<Sdouble> const& values2);

Sdouble minimum(Sdouble const& a, Sdouble const& b);
Sdouble maximum(Sdouble const& a, Sdouble const& b);

double digits(double const& number, double const& error);
std::string numericalNoiseDetailed(Sdouble const& n);
bool numericalNoise(Sdouble const& n);

#endif
