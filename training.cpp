#include "training.h"


std::map<std::string,double> train(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, double const& eps, int const& maxEpoch, double const& learning_rate,
double const& clip, double const& seuil, double const& beta1, double const& beta2, int const& batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(famille_algo=="Classic")
    {
        study = train_classic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,clip,batch_size,beta1,beta2,eps,maxEpoch,
        tracking,track_continuous,record,fileExtension);
    }
    else if(famille_algo=="LM")
    {
        study = train_LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,eps,maxEpoch,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,
        alphaChap,epsDiag,0.1,2.0,3.0,3,
        tracking,track_continuous,record,fileExtension);
    }
    else if(famille_algo=="Perso")
    {
        study = train_Perso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate, beta1,beta2,batch_size, mu,seuil,eps,maxEpoch,
        tracking,track_continuous,record,fileExtension);
    }
    else if(famille_algo=="Incremental")
    {
        study = train_Incremental(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate, beta1,beta2,batch_size, mu,seuil,eps,maxEpoch,
        tracking,track_continuous,record,fileExtension);
    }
    else if(famille_algo=="Essai")
    {
        study = train_Essai(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,beta1,beta2,mu,seuil,eps,maxEpoch,tracking,track_continuous,record,fileExtension);
    }

    return study;

}
