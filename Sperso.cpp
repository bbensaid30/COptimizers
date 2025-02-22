#include "Sperso.h"

std::map<std::string,double> SLCEGD(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SLCEGD"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iter=0, iterLoop=0, total_iterLoop=0, l;
    assert(batch_size<=P);
    bool condition=false;
    int m;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}

    Eigen::MatrixXd echantillonX, echantillonY;
    double learning_rate, costPrec, cost, Vdot, gradientNorm;
    learning_rate=learning_rate_init;
    double const lambda=0.5;
    bool active_other=false;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::shuffle(indices.data(), indices.data()+P, eng);
    echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
    echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
    //deterministic part
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();
    //stochastic part
    As[0]=echantillonX;
    fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
    backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

    while (gradientNorm>eps && epoch<maxEpoch)
    {
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

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec = risk(echantillonY,batch_size,As[L],type_perte); Vdot=gradient.squaredNorm();
        iterLoop=0;
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,batch_size,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*Vdot);
            if(condition)
            {
                learning_rate/=f1;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        total_iterLoop+=iterLoop;

        //std::cout << "eta: " << learning_rate << std::endl;
        //std::cout << "iterLoop: " << iterLoop << std::endl;

        if(learning_rate*gradientNorm<__DBL_EPSILON__){active_other=true; break;}

        learning_rate*=f2;

        std::shuffle(indices.data(), indices.data()+P, eng);
        echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
        echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);

        //deterministic part
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();
        //stochastic part
        As[0]=echantillonX;
        fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        iter++;
        if(iter>0 && iter%m==0){epoch++;}
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,double> study;

    if(active_other)
    {
        std::cout << "LCEGDR activé" << std::endl;
        study = SLCEGDR(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,f1,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }
    else
    {
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        cost = risk(Y,P,As[L],type_perte);
    }

    if(!active_other){
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time); study["total_iterLoop"]=double(total_iterLoop);
    }
    else
    {   
        study["epoch"]+=double(epoch); study["time"]+=double(time); study["total_iterLoop"]+=double(total_iterLoop);
    }

    return study;

}

std::map<std::string,double> SLCEGDR(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SLCEGDR"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iter=0, iterLoop=0, total_iterLoop=0, l;
    assert(batch_size<=P);
    bool condition=false;
    int m;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}

    Eigen::MatrixXd echantillonX, echantillonY;
    double learning_rate, costPrec, cost, Vdot, gradientNorm;
    learning_rate=learning_rate_init;
    double const lambda=0.5;
    bool active_other=false;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::shuffle(indices.data(), indices.data()+P, eng);
    echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
    echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
    //deterministic part
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();
    //stochastic part
    As[0]=echantillonX;
    fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
    backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

    while (gradientNorm>eps && iter<m*maxEpoch)
    {
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

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec = risk(echantillonY,batch_size,As[L],type_perte); Vdot=gradient.squaredNorm();
        iterLoop=0;
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,batch_size,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*Vdot);
            if(condition)
            {
                learning_rate/=f1;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        total_iterLoop+=iterLoop;

        //std::cout << "eta: " << learning_rate << std::endl;
        //std::cout << "iterLoop: " << iterLoop << std::endl;

        if(learning_rate*gradientNorm<__DBL_EPSILON__){active_other=true; break;}

        learning_rate=learning_rate_init;

        std::shuffle(indices.data(), indices.data()+P, eng);
        echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
        echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);

        //deterministic part
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();
        //stochastic part
        As[0]=echantillonX;
        fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        
        iter++;
        if(iter>0 && iter%m==0){epoch++;}
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    
    std::map<std::string,double> study;

    /* if(active_other)
    {
        std::cout << "Adam activé" << std::endl;
        study = SAdam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,1-0.9,1-0.999,batch_size,Sseed,eps,maxEpoch,tracking,record);
    } */

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time); study["total_iterLoop"]=double(total_iterLoop);

    return study;

}

std::map<std::string,double> train_Sperso(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="SLCEGD")
    {
        study = SLCEGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,f1,f2,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="SLCEGDR")
    {
        study = SLCEGDR(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,f1,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }

    return study;

}