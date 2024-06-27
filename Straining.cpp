#include "Straining.h"


std::map<std::string,Sdouble> Strain(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, Sdouble const& eps, int const& maxIter, Sdouble const& learning_rate,
Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, unsigned const Sseed,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(famille_algo=="Sclassic")
    {
        study = train_Sclassic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,batch_size,Sseed,beta1,beta2,eps,maxIter,
        tracking,record,fileExtension);
    }
    else if(famille_algo=="Sperso")
    {
        study = train_Sperso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,batch_size,2,10000,Sseed,eps,maxIter,
        tracking,record,fileExtension);
    }

    return study;

}
