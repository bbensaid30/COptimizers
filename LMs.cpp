#include "LMs.h"

std::map<std::string,double> LM_base(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& mu,
double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_base_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    double prop_entropie=0, prop_initial_ineq=0;

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    double costInit, cost, costPrec;

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

    double gradientNorm = gradient.norm();
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        update(L,nbNeurons,globalIndices,weights,bias,delta); epoch++;

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
            if(!std::signbit((cost-costPrec))){prop_entropie++;}
            if(!std::signbit((cost-costInit))){prop_initial_ineq++;}
        }

        gradientNorm = gradient.norm();

        solve(gradient,Q+mu*I,delta);

    }

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    if(tracking){study["prop_entropie"]=prop_entropie/double(epoch); study["prop_initial_ineq"]=prop_initial_ineq/double(epoch); }

    return study;

}

std::map<std::string,double> LM(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double& mu, double& factor, double const& eps,
int const& maxEpoch, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LM_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LM_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    assert (factor>1);

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    double cost = risk(Y,P,As[L],type_perte), costPrec;
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
    double gradientNorm = gradient.norm();
    while (gradientNorm>eps && epoch<maxEpoch)
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

        if (std::signbit((cost-costPrec)))
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
            endSequence = epoch;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        std::cout << "gradientNorm: " << gradientNorm << std::endl;
        H = Q+mu*I;
        solve(gradient,H,delta);

        epoch++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    endSequence = epoch;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    study["startSequenceMax"]=double(endSequenceMax-notBackMax);
    study["endSequenceMax"]=double(endSequenceMax); study["startSequenceFinal"]=double(epoch-notBack); study["propBack"]=double(nbBack)/double(epoch);

    return study;

}

std::map<std::string,double> LMF(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& eps, int const& maxEpoch,
double const& RMin, double const& RMax, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LMF_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMF_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMF_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    double cost = risk(Y,P,As[L],type_perte), costPrec;
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

    double gradientNorm = gradient.norm();
    while (gradientNorm>eps && epoch<maxEpoch)
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

        if(!std::signbit((R-RMax)))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin)))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2))){factor = 2;}
            if(!std::signbit((factor-10))){factor = 10;}

            if(mu<std::pow(10,-16))
            {
//                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
//                muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }

        if (std::signbit((cost-costPrec)))
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
            endSequence = epoch;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*D;
        solve(gradient,H,delta);

        epoch++;
    }
    endSequence = epoch;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,double> study;
    study["epoch"]=(double)epoch; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(epoch-notBack); study["propBack"]=(double)nbBack/(double)epoch;

    return study;

}

std::map<std::string,double> LMNielson(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& eps, int const& maxEpoch, double const& tau, double const& beta, double const& gamma, int const& p, double const& epsDiag,
bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMNielson_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMNielson_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd I  = Eigen::MatrixXd::Identity(N,N);
    double mu, linearReduction, R, nu=beta, intermed;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    double cost = risk(Y,P,As[L],type_perte), costPrec;
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

    double gradientNorm = gradient.norm();
    while (gradientNorm>eps && epoch<maxEpoch)
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

        if (!std::signbit(R))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            mu*=std::max(1.0/gamma,1-(beta-1)*std::pow(2*R-1,p));
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = epoch;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
            mu*=nu; nu*=2;
        }

        H = Q+mu*I;
        solve(gradient,H,delta);

        epoch++;
    }
    endSequence = epoch;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,double> study;
    study["epoch"]=(double)epoch; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(epoch-notBack); study["propBack"]=(double)nbBack/(double)epoch;

    return study;
}

std::map<std::string,double> LMUphill(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& eps, int const& maxEpoch,
double const& RMin, double const& RMax, int const& b, bool const record, std::string const fileExtension)
{

    assert(b==1 || b==2);

    std::ofstream weightsFlux(("Record/weights_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LMUphill_"+fileExtension+".csv").c_str());
    std::ofstream muFlux(("Record/mu_LMUphill_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}


    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    int endSequence=0, endSequenceMax=0, notBack=1, notBackMax=0, nbBack=0;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N), deltaPrec(N);
    Eigen::MatrixXd H(N,N);
    Eigen::MatrixXd D(N,N);
    double factor, linearReduction, R, mu=10, muc=0, intermed;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    double cost = risk(Y,P,As[L],type_perte), costPrec; double costMin=cost;
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

    double gradientNorm = gradient.norm();
    double deltaNorm = delta.norm();
    double angle;

    while (gradientNorm>eps && epoch<maxEpoch && deltaNorm>eps*std::pow(10,-3))
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

        if(!std::signbit((R-RMax)))
        {
            mu/=2;
            if(mu<muc){mu=0;}
        }
        else if(std::signbit((R-RMin)))
        {
            factor = 2*(costPrec-cost)/(intermed)+2;
            if(std::signbit((factor-2))){factor = 2;}
            if(!std::signbit((factor-10))){factor = 10;}

            if(mu<std::pow(10,-16))
            {
                //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
                //muc = 1.0/(es.eigenvalues().minCoeff());
                muc = 1/(Q.inverse().diagonal().cwiseAbs().maxCoeff());
                mu=muc;
                factor/=2;
            }
            mu*=factor;
        }
        angle = std::pow(1-cosVector(deltaPrec,delta),b)*cost;
        if (std::signbit((cost-costPrec)) || (epoch>1 && std::signbit((angle-costMin))))
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
            gradient.setZero(); Q.setZero();
            notBack++;
            costMin=std::min(costMin,cost);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            gradientNorm = gradient.norm();
        }
        else
        {
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            endSequence = epoch;
            if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}
            notBack=0; nbBack++;
        }

        H = Q+mu*D;
        deltaPrec=delta; solve(gradient,H,delta);

        deltaNorm = delta.norm();
        epoch++;
    }
    endSequence = epoch;
    if (notBack>notBackMax){notBackMax=notBack; endSequenceMax=endSequence;}

    if(record){muFlux << mu << std::endl;}

    std::map<std::string,double> study;
    study["epoch"]=(double)epoch; study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["startSequenceMax"]=(double)(endSequenceMax-notBackMax);
    study["endSequenceMax"]=(double)endSequenceMax; study["startSequenceFinal"]=(double)(epoch-notBack); study["propBack"]=(double)nbBack/(double)epoch;

    return study;

}


std::map<std::string,double> init(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons,std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LM_"+fileExtension+".csv").c_str());
    std::ofstream costFlux(("Record/cost_LM_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !costFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(),l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N,N);
    Eigen::VectorXd delta(N);

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    double cost = risk(Y,P,As[L],type_perte);
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

    std::map<std::string,double> study;
    study["finalGradient"]=gradient.norm(); study["finalCost"]=cost;

    return study;

}

std::map<std::string,double> train_LM(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations,std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
std::string const& algo, double const& eps, int const& maxEpoch, double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag, double const& tau, double const& beta,
double const& gamma, int const& p, bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="LM_base"){study = LM_base(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu,eps,maxEpoch,tracking,track_continuous,record,fileExtension);}
    else if(algo=="LM"){study = LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu,factor,eps,maxEpoch,record,fileExtension);}
    else if(algo=="LMF"){study = LMF(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,eps,maxEpoch,RMin,RMax,record,fileExtension);}
    else if(algo=="LMUphill"){study = LMUphill(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,eps,maxEpoch,RMin,RMax,b,record,fileExtension);}
    else if(algo=="LMNielson"){study = LMNielson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,eps,maxEpoch,tau,beta,gamma,p,epsDiag,record,fileExtension);}
    else if(algo=="init"){study = init(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,record,fileExtension);}

    return study;
}
