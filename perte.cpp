#include "perte.h"

//--------------------------------------------------------------- Calcul de L ------------------------------------------------------------------------------------------------------------------

Sdouble norme2(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y)
{
    return 0.5*(x-y).squaredNorm();
}

Sdouble difference(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y)
{
    return (x-y).sum();
}

Sdouble entropie_generale(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y)
{
    return -(x.array()*(y.array().log())).sum();
}

Sdouble entropie_one(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y)
{
    assert(x.rows()==1);
    return -x(0)*Sstd::log(y(0))-(1-x(0))*Sstd::log(1-y(0));
}

Sdouble KL_divergence(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y)
{
    return (x.array()*((x.array()*y.cwiseInverse().array()).log())).sum();
}

Sdouble L(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, std::string type_perte)
{
    if(type_perte == "norme2"){return norme2(x,y);}
    else if (type_perte == "difference"){return difference(x,y);}
    else if (type_perte == "entropie_generale"){return entropie_generale(x,y);}
    else if(type_perte == "entropie_one"){return entropie_one(x,y);}
    else if(type_perte == "KL_divergence"){return KL_divergence(x,y);}
}
//------------------------------------------------------ Calcul de L' -------------------------------------------------------------------------------------------------------------------------------------


void FO_norme2(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP)
{
    LP=y-x;
}

void FO_difference(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP)
{
    int const taille = x.rows();
    LP = Eigen::SVectorXd::Constant(taille,-1);
}

void FO_entropie_generale(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP)
{
    LP=-x.cwiseProduct(y.cwiseInverse());
}

void FO_entropie_one(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP)
{
    assert(x.rows()==1);
    LP = -x.array()/y.array()+(1-x.array())/(1-y.array());
}

void FO_KL_divergence(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP)
{
    LP=-x.cwiseProduct(y.cwiseInverse());
}

void FO_L(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, std::string type_perte)
{
    if(type_perte == "norme2"){FO_norme2(x,y,LP);}
    else if (type_perte == "difference"){FO_difference(x,y,LP);}
    else if (type_perte == "entropie_generale"){FO_entropie_generale(x,y,LP);}
    else if(type_perte == "entropie_one"){FO_entropie_one(x,y,LP);}
    else if(type_perte == "KL_divergence"){FO_KL_divergence(x,y,LP);}
}

//-------------------------------------------------------------------Calcul de L' et L'' -----------------------------------------------------------------------------------------------------------------------

void SO_norme2(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP)
{
    int const taille = x.rows();
    LP = y-x;
    LPP = Eigen::SMatrixXd::Identity(taille,taille);
}

void SO_difference(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP)
{
    int const taille = x.rows();
    LP = Eigen::SVectorXd::Constant(taille,-1);
    LPP = Eigen::SMatrixXd::Zero(taille,taille);
}

void SO_entropie_generale(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP)
{
    int const taille = x.rows();
    LP=-x.cwiseProduct(y.cwiseInverse());
    LPP = Eigen::SMatrixXd::Zero(taille,taille);
    for(int i=0; i<taille; i++)
    {
        LPP(i,i) = x(i)/Sstd::pow(y(i),2);
    }
}
void SO_entropie_one(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP)
{
    assert(x.rows()==1);
    LP = -x.array()/y.array()+(1-x.array())/(1-y.array());
    LPP = x.array()/y.array().pow(2)+(1-x.array())/(1-y.array()).pow(2);
}

void SO_KL_divergence(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP)
{
    int const taille = x.rows();
    LP=-x.cwiseProduct(y.cwiseInverse());
    LPP = Eigen::SMatrixXd::Zero(taille,taille);
    for(int i=0; i<taille; i++)
    {
        LPP(i,i) = x(i)/Sstd::pow(y(i),2);
    }
}

void SO_L(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP, std::string type_perte)
{
    if(type_perte == "norme2"){SO_norme2(x,y,LP,LPP);}
    else if (type_perte == "difference"){SO_difference(x,y,LP,LPP);}
    else if (type_perte == "entropie_generale"){SO_entropie_generale(x,y,LP,LPP);}
    else if(type_perte == "entropie_one"){SO_entropie_one(x,y,LP,LPP);}
    else if(type_perte == "KL_divergence"){SO_KL_divergence(x,y,LP,LPP);}
}

