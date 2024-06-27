#ifndef SCALING
#define SCALING

#include <Eigen/Dense>

void scalingFletcher(Eigen::MatrixXd const& Q, Eigen::MatrixXd& D, int const& N);
void scalingMore(Eigen::VectorXd const& gradient, Eigen::MatrixXd& D);
void scalingMore2(Eigen::MatrixXd const& Q, Eigen::MatrixXd& D, double const& epsDiag);
void scalingFletcherMore(Eigen::MatrixXd const& Q, Eigen::MatrixXd& D, int const& N);

#endif
