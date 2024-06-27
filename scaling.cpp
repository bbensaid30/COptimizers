#include "scaling.h"

void scalingFletcher(Eigen::MatrixXd const& Q, Eigen::MatrixXd& D, int const& N)
{
    D = Q.diagonal().asDiagonal();
    for(int i=0;i<N;i++)
    {
        if(D(i,i)<std::pow(10,-16)){D(i,i)=double(1);}
    }
}

void scalingMore(Eigen::VectorXd const& gradient, Eigen::MatrixXd& D)
{
    D = D.cwiseMax(Eigen::MatrixXd(gradient.cwiseAbs().asDiagonal()));
}
void scalingMore2(Eigen::MatrixXd const& Q, Eigen::MatrixXd& D, double const& epsDiag)
{
    D = D.cwiseMax(Eigen::MatrixXd(Q.diagonal().cwiseAbs().cwiseMax(epsDiag).asDiagonal()));
}

void scalingFletcherMore(Eigen::MatrixXd const& Q, Eigen::MatrixXd& D, int const& N)
{
    D = D.cwiseMax(Eigen::MatrixXd(Q.diagonal().cwiseAbs().asDiagonal()));
    for(int i=0;i<N;i++)
    {
        if(D(i,i)<std::pow(10,-16)){D(i,i)=1;}
    }
}


