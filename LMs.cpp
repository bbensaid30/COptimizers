#include "LMs.h"

std::map<std::string,Sdouble> LM_base(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& mu,
Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_base_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    Sdouble prop_entropie=0, prop_initial_ineq=0;

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);

    Sdouble costInit, cost, costPrec;

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
    }
    solve(gradient,Q+mu*I,delta);

    if(tracking)
    {
        cost = risk(Y,P,As[L],type_perte);
        costInit = cost;
    }

    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        update(L,nbNeurons,globalIndices,weights,bias,delta); iter++;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);

        if(tracking)
        {
            costPrec = cost;
            cost = risk(Y,P,As[L],type_perte);
            if(!std::signbit((cost-costPrec).number)){prop_entropie++;}
            if(!std::signbit((cost-costInit).number)){prop_initial_ineq++;}
        }

        gradientNorm = gradient.norm();
        if(numericalNoise(gradientNorm)){break;}

        solve(gradient,Q+mu*I,delta);

    }

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    if(tracking){study["prop_entropie"]=prop_entropie/Sdouble(iter); study["prop_initial_ineq"]=prop_initial_ineq/Sdouble(iter); }

    return study;

}

std::map<std::string,Sdouble> LM(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble& mu, Sdouble& factor, Sdouble const& eps,
int const& maxIter, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LM_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LM_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte), costPrec;
    QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
    H = Q+mu*I;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        update(L,nbNeurons,globalIndices,weights,bias,delta);
        costPrec=cost;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if (std::signbit((cost-costPrec).number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            mu/=factor;
            notBack++;
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            mu*=factor;
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu)){break;}

        std::cout << "gradientNorm: " << gradientNorm << std::endl;
        H = Q+mu*I;
        solve(gradient,H,delta);

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    study["startSequenceMax"]=Sdouble(endSequenceMax-notBackMax);
    study["endSequenceMax"]=Sdouble(endSequenceMax); study["startSequenceFinal"]=Sdouble(iter-notBack); study["propBack"]=Sdouble(nbBack)/Sdouble(iter);

    return study;

}

std::map<std::string,Sdouble> LMF(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LMF_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMF_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMF_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte), costPrec;
    QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
    scalingFletcher(Q,D,N);
    H=Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);

    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        update(L,nbNeurons,globalIndices,weights,bias,delta);
        costPrec = cost;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);
        intermed = delta.transpose()*gradient;
        linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
             costFlux << cost << std::endl;
             muFlux << mu << std::endl;
        }

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2).number)){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
//                Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
//                muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }

        if (std::signbit((cost-costPrec).number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

        H = Q+mu*D;
        solve(gradient,H,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;

}

std::map<std::string,Sdouble> LMNielson(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& eps, int const& maxIter, Sdouble const& tau, Sdouble const& beta, Sdouble const& gamma, int const& p, Sdouble const& epsDiag,
bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMNielson_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd I  = Eigen::SMatrixXd::Identity(N,N);
    Sdouble mu, linearReduction, R, nu=beta, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte), costPrec;
    QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
    mu=tau*Q.diagonal().maxCoeff();
    H = Q+mu*I;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);

    Sdouble gradientNorm = gradient.norm();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        update(L,nbNeurons,globalIndices,weights,bias,delta);
        costPrec = cost;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);
        intermed = delta.transpose()*gradient;
        linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if (!std::signbit(R.number))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            mu*=maximum(1.0/gamma,1-(beta-1)*Sstd::pow(2*R-1,p));
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
            mu*=nu; nu*=2;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(nu)){break;}

        H = Q+mu*I;
        solve(gradient,H,delta);

        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;
}

std::map<std::string,Sdouble> LMUphill(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& eps, int const& maxIter,
Sdouble const& RMin, Sdouble const& RMax, int const& b, bool const record, std::string const fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMUphill_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N), deltaPrec(N);
    Eigen::SMatrixXd H(N,N);
    Eigen::SMatrixXd D(N,N);
    Sdouble factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte), costPrec; Sdouble costMin=cost;
    QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
    scalingFletcher(Q,D,N);
    H = Q+mu*D;

    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }
    solve(gradient,H,delta);

    Sdouble gradientNorm = gradient.norm();
    Sdouble deltaNorm = delta.norm();
    Sdouble angle;

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter && deltaNorm+std::abs(deltaNorm.error)>eps*std::pow(10,-3))
    {
        update(L,nbNeurons,globalIndices,weights,bias,delta);
        costPrec = cost;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);
        intermed = delta.transpose()*gradient;
        linearReduction = -delta.transpose()*Q*delta; linearReduction-=2*intermed;
        R = 2*(costPrec-cost)/linearReduction;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
            costFlux << cost << std::endl;
            muFlux << mu << std::endl;
        }

        if(!std::signbit((R-RMax).number))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin).number))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2).number)){factor = 2;}
            if(!std::signbit((factor-10).number)){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::SMatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        angle = Sstd::pow(1-cosVector(deltaPrec,delta),b)*cost;
        if (std::signbit((cost-costPrec).number) || (iter>1 && std::signbit((angle-costMin).number)))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            costMin=minimum(costMin,cost);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = iter;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        if(numericalNoise(cost) || numericalNoise(mu) || numericalNoise(factor)){break;}

        H = Q+mu*D;
        deltaPrec=delta; solve(gradient,H,delta);

        deltaNorm = delta.norm();
        iter++;
    }
    endSequence = iter;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,Sdouble> study;
    study["iter"]=(Sdouble)iter; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(Sdouble)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(Sdouble)endSequenceMax; study["startSequenceFinal"]=(Sdouble)(iter-notBack); study["propBack"]=(Sdouble)nbBack/(Sdouble)iter;

    return study;

}


std::map<std::string,Sdouble> init(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons,std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LM_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(),l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SMatrixXd Q = Eigen::SMatrixXd::Zero(N,N);
    Eigen::SVectorXd delta(N);

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    Sdouble cost = risk(Y,P,As[L],type_perte);
    QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }
        costFlux << cost << std::endl;
    }

    std::map<std::string,Sdouble> study;
    study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    return study;

}

std::map<std::string,Sdouble> train_LM(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
std::string const& algo, Sdouble const& eps, int const& maxIter, Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag, Sdouble const& tau, Sdouble const& beta,
Sdouble const& gamma, int const& p, bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="LM_base"){study = LM_base(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu,eps,maxIter,tracking,track_continuous,record,fileExtension);}
    else if(algo=="LM"){study = LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu,factor,eps,maxIter,record,fileExtension);}
    else if(algo=="LMF"){study = LMF(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,eps,maxIter,RMin,RMax,record,fileExtension);}
    else if(algo=="LMUphill"){study = LMUphill(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,eps,maxIter,RMin,RMax,b,record,fileExtension);}
    else if(algo=="LMNielson"){study = LMNielson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,eps,maxIter,tau,beta,gamma,p,epsDiag,record,fileExtension);}
    else if(algo=="init"){study = init(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,record,fileExtension);}

    return study;
}
