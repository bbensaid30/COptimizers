#include "training.h"


std::map<std::string,Sdouble> train(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, Sdouble const& eps, int const& maxIter, Sdouble const& learning_rate,
Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(famille_algo=="Classic")
    {
        study = train_classic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,clip,batch_size,beta1,beta2,eps,maxIter,
        tracking,track_continuous,record,fileExtension);
    }
    else if(famille_algo=="LM")
    {
        study = train_LM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,eps,maxIter,mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,
        alphaChap,epsDiag,0.1,2.0,3.0,3,
        tracking,track_continuous,record,fileExtension);
    }
    else if(famille_algo=="Perso")
    {
        study = train_Perso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate, beta1,beta2,batch_size, mu,seuil,eps,maxIter,
        tracking,track_continuous,record,fileExtension);
    }
    else if(famille_algo=="Essai")
    {
        study = train_Essai(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,beta1,beta2,mu,seuil,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }

    return study;

}
