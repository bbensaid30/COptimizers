#include "activations.h"

void linear(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    S = Eigen::SMatrixXd::Constant(Z.rows(),Z.cols(),1);
}

void sigmoid(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Z = (1+(-Z).array().exp()).inverse();
    S = Z.array()*(1-Z.array());
}

//approximation de GELU
void GELU(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Sdouble const c = 1.702;

    Eigen::SMatrixXd U = Z;
    Z = (1+(-c*Z).array().exp()).inverse();
    S = Z.array()*(1+c*U.array()*(1-Z.array()));
    Z = U.array()*Z.array();
}

void softmax(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, int const q)
{
    int const tailleX=Z.rows(), tailleY=Z.cols();
    assert(q>=-1); assert(q<tailleX);
    Eigen::SMatrixXd A(tailleX,tailleY);

    A = Z.array().exp();
    Eigen::SVectorXd sumExp = A.colwise().sum().cwiseInverse();
    A = A * sumExp.asDiagonal();

    if(q==-1){S=Z;}
    else
    {
        for(int r=0; r<tailleX; r++)
        {
            for(int p=0; p<tailleY; p++)
            {
                if (q==r){S(r,p) = A(r,p)*(1-A(r,p));}
                else{S(r,p) = -Sstd::exp(Z(r,p)-Z(q,p))*Sstd::pow(A(q,p),2);}
            }
        }
    }

    Z=A;

}

void tanh(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Z = Z.array().tanh();
    S = 1-Z.array().pow(2);
}

void reLU(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Eigen::SMatrixXd U = Eigen::SMatrixXd::Constant(Z.rows(),Z.cols(),1);
    Z = Z.cwiseMax(0);
    S = (Z.array()>0).select(U,0);
}

void softplus(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    S = (1+(-Z).array().exp()).inverse();
    Z = (1+Z.array().exp()).log();
}

void IDC(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    S = (Z.array())*(2*(1+Z.array().pow(2)).sqrt()).inverse()+1;
    Z = Z.array()+((1+Z.array().pow(2)).sqrt()-1)/2;
}

void sinus(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    S = Z.array().cos();
    Z = Z.array().sin();
}


void activation(std::string nameActivation, Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, int const q)
{
    if (nameActivation == "sigmoid")
    {
        sigmoid(Z,S);
    }
    else if (nameActivation == "linear")
    {
        linear(Z,S);
    }
    else if(nameActivation == "tanh")
    {
        tanh(Z,S);
    }
    else if(nameActivation == "reLU")
    {
        reLU(Z,S);
    }
    else if(nameActivation == "GELU")
    {
        GELU(Z,S);
    }
    else if(nameActivation == "softplus")
    {
        softplus(Z,S);
    }
    else if(nameActivation == "IDC")
    {
        IDC(Z,S);
    }
    else if(nameActivation == "sinus")
    {
        sinus(Z,S);
    }
    else if(nameActivation == "softmax")
    {
        softmax(Z,S,q);
    }



    else if(nameActivation == "polyTwo")
    {
        polyTwo(Z,S);
    }
    else if(nameActivation=="polyThree")
    {
        polyThree(Z,S);
    }
    else if(nameActivation=="polyFour")
    {
        polyFour(Z,S);
    }
    else if(nameActivation=="polyFive")
    {
        polyFive(Z,S);
    }
    else if(nameActivation=="polyEight")
    {
        polyEight(Z,S);
    }
    else if(nameActivation=="expTwo")
    {
        expTwo(Z,S);
    }
    else if(nameActivation=="expFour")
    {
        expFour(Z,S);
    }
    else if(nameActivation=="expFive")
    {
        expFive(Z,S);
    }
    else if(nameActivation=="ratTwo")
    {
        ratTwo(Z,S,1);
    }
    else if(nameActivation=="cloche")
    {
        cloche(Z,S);
    }
    else
    {
        linear(Z,S);
    }
}


void polyTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    S = 2*Sstd::exp(c/2)*Z.array();
    Z = Sstd::exp(c/2)*(Z.array().pow(2)-1);
}

void polyThree(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    S = 6*Sstd::exp(c/2)*Z.array()*(Z.array()-1);
    Z = Sstd::exp(c/2)*(2*Z.array().pow(3)-3*Z.array().pow(2)+5);
}

void polyFour(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    S = 4*Sstd::exp(c/2)*Z.array()*(Z.array().pow(2)-1);
    Z = Sstd::exp(c/2)*(Z.array().pow(4)-2*Z.array().pow(2)+3);
}

void polyFive(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    S = 5*Z.array().pow(4)-16*Z.array().pow(3)+6*Z.array().pow(2)+16*Z.array()-11;
    Z = Z.array().pow(5)-4*Z.array().pow(4)+2*Z.array().pow(3)+8*Z.array().pow(2)-11*Z.array()-12;
}

void polyEight(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    S = 280*Z.array()*(Z.array()-1).pow(3)*(Z.array()-2).pow(3);
    Z = 35*Z.array().pow(8)-360*Z.array().pow(7)+1540*Z.array().pow(6)-3528*Z.array().pow(5)+4620*Z.array().pow(4)-3360*Z.array().pow(3)+1120*Z.array().pow(2)+1;
}

void expTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Eigen::SMatrixXd inter = (1/2*(Z.array().pow(2)-2*Z.array().pow(2)+c)).exp();
    S = (Z.array()-1)*inter.array();
    Z = inter;
}

void expFour(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Eigen::SMatrixXd inter = (1/2*(Z.array().pow(4)-2*Z.array().pow(2)+c)).exp();
    S = 2*Z.array()*(Z.array().pow(2)-1)*inter.array();
    Z = inter;
}

void expFive(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Eigen::SMatrixXd inter = (1/2*(6*Z.array().pow(5)-15*Z.array().pow(4)-10*Z.array().pow(3)+30*Z.array().pow(2)+c)).exp();
    S = 15*Z.array()*(Z.array().pow(3)-2*Z.array().pow(2)-Z.array()+2)*inter.array();
    Z = inter;
}

void ratTwo(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S, Sdouble c)
{
    Sdouble const normalization = 2.0/(c+2+Sstd::sqrt(c*c+4));
    Eigen::SMatrixXd poly = Z.array().pow(2)-2*Z.array()+2;
    S = (-2*Z.array().pow(2)+2*(2-c)*Z.array()+2*c)*(poly.array().pow(2)).inverse(); S*=normalization;
    Z = (Z.array().pow(2)+c)*poly.array().inverse(); Z*=normalization;
}

void cloche(Eigen::SMatrixXd& Z, Eigen::SMatrixXd& S)
{
    Eigen::SMatrixXd exp2 = (-0.5*Z.array().pow(2)).exp();
    S = -Z.array()*exp2.array();
    Z = exp2;
}
