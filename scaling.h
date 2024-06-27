#ifndef SCALING
#define SCALING

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

void scalingFletcher(Eigen::SMatrixXd const& Q, Eigen::SMatrixXd& D, int const& N);
void scalingMore(Eigen::SVectorXd const& gradient, Eigen::SMatrixXd& D);
void scalingMore2(Eigen::SMatrixXd const& Q, Eigen::SMatrixXd& D, Sdouble const& epsDiag);
void scalingFletcherMore(Eigen::SMatrixXd const& Q, Eigen::SMatrixXd& D, int const& N);

#endif
