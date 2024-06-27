#include "propagation.h"

//------------------------------------------------------------------ Propagation directe ----------------------------------------------------------------------------------------

void fforward(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes)
{
    int l;
    for (l=0;l<L;l++)
    {
        As[l+1] = weights[l]*As[l];
        As[l+1].colwise() += bias[l];
        activation(activations[l], As[l+1], slopes[l]);
    }
}

double risk(Eigen::MatrixXd const& Y, int const& P, Eigen::MatrixXd const& output_network, std::string const& type_perte, bool const normalized)
{
    double cost=0;
    for(int p=0; p<P; p++)
    {
        cost+=L(Y.col(p),output_network.col(p),type_perte);
    }
    if(normalized){cost/=(double)P;}
    return cost;
}

//------------------------------------------------------------------ Rétropropagation ---------------------------------------------------------------------------------------------

void backward(Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::VectorXd& gradient, std::string const& type_perte, bool const normalized)
{
    int l,p,n,nL=nbNeurons[L],jump;
    int N=globalIndices[2*L-1];

    Eigen::MatrixXd L_derivative(nL,P);
    Eigen::VectorXd LP(nL);

    Eigen::MatrixXd dzL(nL,P);
    Eigen::MatrixXd dz;
    Eigen::MatrixXd dw;
    Eigen::VectorXd db;

    //#pragma omp parallel for private(LP)
    for(p=0; p<P; p++)
    {
        FO_L(Y.col(p),As[L].col(p),LP,type_perte);
        L_derivative.col(p)=LP;
    }
    //#pragma omp barrier

    if(activations[L-1]=="softmax")
    {
        dzL.setZero();
        Eigen::MatrixXd ZL(nL,P);

        ZL = slopes[L-1];

        for(int r=0; r<nL; r++)
        {
            activation(activations[L-1],ZL,slopes[L-1],r);

            dzL.row(r) = (L_derivative.cwiseProduct(slopes[L-1])).colwise().sum();
        }

    }
    else{dzL = L_derivative.cwiseProduct(slopes[L-1]);}

    dz=dzL;
    jump=nbNeurons[L]*nbNeurons[L-1];
    dw = dz*(As[L-1].transpose());
    db = dz.rowwise().sum();
    dw.resize(jump,1);
    gradient.segment(globalIndices[2*(L-1)]-jump,jump)=dw;
    jump=nbNeurons[L];
    gradient.segment(globalIndices[2*(L-1)+1]-jump,jump)=db;
    for (l=L-1;l>0;l--)
    {
        dz=(weights[l].transpose()*dz).cwiseProduct(slopes[l-1]);

        jump=nbNeurons[l]*nbNeurons[l-1];
        dw=dz*(As[l-1].transpose());
        db = dz.rowwise().sum();
        dw.resize(jump,1);
        gradient.segment(globalIndices[2*(l-1)]-jump,jump)=dw;
        jump=nbNeurons[l];
        gradient.segment(globalIndices[2*(l-1)+1]-jump,jump)=db;
    }
    if(normalized){gradient/=(double)P;}
}


//Cas où L'' diagonale (la plupart du temps)
void QSO_backward(Eigen::MatrixXd const& Y, int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations,
std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes,
Eigen::VectorXd& gradient, Eigen::MatrixXd& Q, std::string const& type_perte)
{
    int l,m,p,n,nL=nbNeurons[L],jump;
    int N=globalIndices[2*L-1];

    Eigen::VectorXd LP(nL);
    Eigen::MatrixXd LPP(nL,nL);
    Eigen::VectorXd dzL(nL);
    Eigen::VectorXd dz;
    Eigen::MatrixXd dw;
    Eigen::VectorXd Jpm(N);

    Eigen::MatrixXd ZL(nL,P);
    if(activations[L-1]=="softmax")
    {
        ZL = slopes[L-1];
    }

    gradient.setZero(); Q.setZero();

    for (p=0;p<P;p++)
    {
        SO_L(Y.col(p),As[L].col(p),LP,LPP,type_perte);
        for (m=0;m<nL;m++)
        {
            if(activations[L-1]=="softmax")
            {
                for (n=0;n<nL;n++)
                {
                    if(m==n){dzL(n) = -As[L](m,p)*(1-As[L](m,p));}
                    else{dzL(n) = std::exp(ZL(n,p)-ZL(m,p))*std::pow(As[L](m,p),2);}
                }
            }
            else
            {
                for (n=0;n<nL;n++)
                {
                    dzL(n) = (n==m) ? -slopes[L-1](m,p) : 0;
                }
            }
            dz=dzL;

            jump=nbNeurons[L]*nbNeurons[L-1];
            dw=dz*(As[L-1].col(p).transpose());
            dw.resize(jump,1);
            Jpm.segment(globalIndices[2*(L-1)]-jump,jump)=dw;
            jump=nbNeurons[L];
            Jpm.segment(globalIndices[2*(L-1)+1]-jump,jump)=dz;
            for (l=L-1;l>0;l--)
            {
                dz=(weights[l].transpose()*dz).cwiseProduct(slopes[l-1].col(p));

                jump=nbNeurons[l]*nbNeurons[l-1];
                dw=dz*(As[l-1].col(p).transpose());
                dw.resize(jump,1);
                Jpm.segment(globalIndices[2*(l-1)]-jump,jump)=dw;
                jump=nbNeurons[l];
                Jpm.segment(globalIndices[2*(l-1)+1]-jump,jump)=dz;
            }
            Q+=LPP(m,m)*Jpm*Jpm.transpose();
            gradient+=-LP(m)*Jpm;
        }
    }
    Q/=(double)P;
    gradient/=(double)P;

}

void QSO_backwardJacob(int const& L, int const& P, std::vector<int> const& nbNeurons, std::vector<std::string> const& activations, std::vector<int> const& globalIndices,
std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::vector<Eigen::MatrixXd>& As, std::vector<Eigen::MatrixXd>& slopes, Eigen::MatrixXd& J)
{
    int l,m,p,n,nL=nbNeurons[L],jump,nbLine=0;
    int N=globalIndices[2*L-1];

    Eigen::VectorXd dzL(nL);
    Eigen::VectorXd dz;
    Eigen::MatrixXd dw;
    Eigen::VectorXd Jpm(N);

    Eigen::MatrixXd ZL(nL,P);
    if(activations[L-1]=="softmax")
    {
        ZL = slopes[L-1];
    }

    for (p=0;p<P;p++)
    {
        for (m=0;m<nL;m++)
        {
            if(activations[L-1]=="softmax")
            {
                for (n=0;n<nL;n++)
                {
                    if(m==n){dzL(n) = -As[L](m,p)*(1-As[L](m,p));}
                    else{dzL(n) = std::exp(ZL(n,p)-ZL(m,p))*std::pow(As[L](m,p),2);}
                }
            }
            else
            {
                for (n=0;n<nL;n++)
                {
                    dzL(n) = (n==m) ? -slopes[L-1](m,p) : 0;
                }
            }
            dz=dzL;

            jump=nbNeurons[L]*nbNeurons[L-1];
            dw=dz*(As[L-1].col(p).transpose());
            dw.resize(jump,1);
            Jpm.segment(globalIndices[2*(L-1)]-jump,jump)=dw;
            jump=nbNeurons[L];
            Jpm.segment(globalIndices[2*(L-1)+1]-jump,jump)=dz;
            for (l=L-1;l>0;l--)
            {
                dz=(weights[l].transpose()*dz).cwiseProduct(slopes[l-1].col(p));

                jump=nbNeurons[l]*nbNeurons[l-1];
                dw=dz*(As[l-1].col(p).transpose());
                dw.resize(jump,1);
                Jpm.segment(globalIndices[2*(l-1)]-jump,jump)=dw;
                jump=nbNeurons[l];
                Jpm.segment(globalIndices[2*(l-1)+1]-jump,jump)=dz;
            }
            J.row(nbLine) = Jpm;
            nbLine++;
        }
    }

}

void solve(Eigen::VectorXd const& gradient, Eigen::MatrixXd const& hessian, Eigen::VectorXd& delta, std::string const method)
{

    if(method=="LLT"){Eigen::LLT<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="LDLT"){Eigen::LDLT<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient); }
    else if(method=="HouseholderQR"){Eigen::HouseholderQR<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="ColPivHouseholderQR"){Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="FullPivHouseholderQR"){Eigen::FullPivHouseholderQR<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="PartialPivLU"){Eigen::PartialPivLU<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="FullPivLU"){Eigen::FullPivLU<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="ConjugateGradient"){Eigen::ConjugateGradient<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="LeastSquaresConjugateGradient"){Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
    else if(method=="BiCGSTAB"){Eigen::BiCGSTAB<Eigen::MatrixXd> solver; solver.compute(hessian); delta = solver.solve(-gradient);}
}

void update(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
Eigen::VectorXd const& delta)
{

    int l, jump;

    for (l=0;l<L;l++)
    {
        jump=nbNeurons[l]*nbNeurons[l+1];
        weights[l].resize(jump,1);
        weights[l] += delta.segment(globalIndices[2*l]-jump,jump);
        weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
        jump=nbNeurons[l+1];
        bias[l] += delta.segment(globalIndices[2*l+1]-jump,jump);
    }

}

void updateNesterov(int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias,
std::vector<Eigen::MatrixXd>& weights2, std::vector<Eigen::VectorXd>& bias2, Eigen::VectorXd const& delta, double const& lambda1, double const& lambda2)
{

    int l, jump;

    for (l=0;l<L;l++)
    {
        jump=nbNeurons[l]*nbNeurons[l+1];
        weights[l].resize(jump,1); weights2[l].resize(jump,1);
        weights[l] = lambda1*weights2[l] + lambda2*delta.segment(globalIndices[2*l]-jump,jump);
        weights[l].resize(nbNeurons[l+1],nbNeurons[l]); weights2[l].resize(nbNeurons[l+1],nbNeurons[l]);
        jump=nbNeurons[l+1];
        bias[l] = lambda1*bias2[l] + lambda2*delta.segment(globalIndices[2*l+1]-jump,jump);
    }

}
