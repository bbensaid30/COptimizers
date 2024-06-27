#include "perte.h"

//--------------------------------------------------------------- Calcul de L ------------------------------------------------------------------------------------------------------------------

double norme2(Eigen::VectorXd const& x, Eigen::VectorXd const& y)
{
    return 0.5*(x-y).squaredNorm();
}

double difference(Eigen::VectorXd const& x, Eigen::VectorXd const& y)
{
    return (x-y).sum();
}

double entropie_generale(Eigen::VectorXd const& x, Eigen::VectorXd const& y)
{
    return -(x.array()*(y.array().log())).sum();
}

double entropie_one(Eigen::VectorXd const& x, Eigen::VectorXd const& y)
{
    assert(x.rows()==1);
    return -x(0)*std::log(y(0))-(1-x(0))*std::log(1-y(0));
}

double KL_divergence(Eigen::VectorXd const& x, Eigen::VectorXd const& y)
{
    return (x.array()*((x.array()*y.cwiseInverse().array()).log())).sum();
}

double L(Eigen::VectorXd const& x, Eigen::VectorXd const& y, std::string type_perte)
{
    if(type_perte == "norme2"){return norme2(x,y);}
    else if (type_perte == "difference"){return difference(x,y);}
    else if (type_perte == "entropie_generale"){return entropie_generale(x,y);}
    else if(type_perte == "entropie_one"){return entropie_one(x,y);}
    else if(type_perte == "KL_divergence"){return KL_divergence(x,y);}
    else{return norme2(x,y);}
}
//------------------------------------------------------ Calcul de L' -------------------------------------------------------------------------------------------------------------------------------------


void FO_norme2(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP)
{
    LP=y-x;
}

void FO_difference(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP)
{
    int const taille = x.rows();
    LP = Eigen::VectorXd::Constant(taille,-1);
}

void FO_entropie_generale(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP)
{
    LP=-x.cwiseProduct(y.cwiseInverse());
}

void FO_entropie_one(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP)
{
    assert(x.rows()==1);
    LP = -x.array()/y.array()+(1-x.array())/(1-y.array());
}

void FO_KL_divergence(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP)
{
    LP=-x.cwiseProduct(y.cwiseInverse());
}

void FO_L(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, std::string type_perte)
{
    if(type_perte == "norme2"){FO_norme2(x,y,LP);}
    else if (type_perte == "difference"){FO_difference(x,y,LP);}
    else if (type_perte == "entropie_generale"){FO_entropie_generale(x,y,LP);}
    else if(type_perte == "entropie_one"){FO_entropie_one(x,y,LP);}
    else if(type_perte == "KL_divergence"){FO_KL_divergence(x,y,LP);}
    else{FO_norme2(x,y,LP);}
}

//-------------------------------------------------------------------Calcul de L' et L'' -----------------------------------------------------------------------------------------------------------------------

void SO_norme2(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP)
{
    int const taille = x.rows();
    LP = y-x;
    LPP = Eigen::MatrixXd::Identity(taille,taille);
}

void SO_difference(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP)
{
    int const taille = x.rows();
    LP = Eigen::VectorXd::Constant(taille,-1);
    LPP = Eigen::MatrixXd::Zero(taille,taille);
}

void SO_entropie_generale(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP)
{
    int const taille = x.rows();
    LP=-x.cwiseProduct(y.cwiseInverse());
    LPP = Eigen::MatrixXd::Zero(taille,taille);
    for(int i=0; i<taille; i++)
    {
        LPP(i,i) = x(i)/std::pow(y(i),2);
    }
}
void SO_entropie_one(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP)
{
    assert(x.rows()==1);
    LP = -x.array()/y.array()+(1-x.array())/(1-y.array());
    LPP = x.array()/y.array().pow(2)+(1-x.array())/(1-y.array()).pow(2);
}

void SO_KL_divergence(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP)
{
    int const taille = x.rows();
    LP=-x.cwiseProduct(y.cwiseInverse());
    LPP = Eigen::MatrixXd::Zero(taille,taille);
    for(int i=0; i<taille; i++)
    {
        LPP(i,i) = x(i)/std::pow(y(i),2);
    }
}

void SO_L(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP, std::string type_perte)
{
    if(type_perte == "norme2"){SO_norme2(x,y,LP,LPP);}
    else if (type_perte == "difference"){SO_difference(x,y,LP,LPP);}
    else if (type_perte == "entropie_generale"){SO_entropie_generale(x,y,LP,LPP);}
    else if(type_perte == "entropie_one"){SO_entropie_one(x,y,LP,LPP);}
    else if(type_perte == "KL_divergence"){SO_KL_divergence(x,y,LP,LPP);}
    else{SO_norme2(x,y,LP,LPP);}
}

