#include "scaling.h"

void scalingFletcher(Eigen::SMatrixXd const& Q, Eigen::SMatrixXd& D, int const& N)
{
    D = Q.diagonal().asDiagonal();
    for(int i=0;i<N;i++)
    {
        if(D(i,i)+std::abs(D(i,i).error)<std::pow(10,-16)){D(i,i)=Sdouble(1);}
    }
}

void scalingMore(Eigen::SVectorXd const& gradient, Eigen::SMatrixXd& D)
{
    D = D.cwiseMax(Eigen::SMatrixXd(gradient.cwiseAbs().asDiagonal()));
}
void scalingMore2(Eigen::SMatrixXd const& Q, Eigen::SMatrixXd& D, Sdouble const& epsDiag)
{
    D = D.cwiseMax(Eigen::SMatrixXd(Q.diagonal().cwiseAbs().cwiseMax(epsDiag).asDiagonal()));
}

void scalingFletcherMore(Eigen::SMatrixXd const& Q, Eigen::SMatrixXd& D, int const& N)
{
    D = D.cwiseMax(Eigen::SMatrixXd(Q.diagonal().cwiseAbs().asDiagonal()));
    for(int i=0;i<N;i++)
    {
        if(D(i,i)+std::abs(D(i,i).error)<std::pow(10,-16)){D(i,i)=1;}
    }
}


