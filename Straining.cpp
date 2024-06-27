#include "Straining.h"


std::map<std::string,double> Strain(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, double const& eps, int const& maxEpoch, double const& learning_rate,
double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(famille_algo=="Sclassic")
    {
        study = train_Sclassic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,batch_size,Sseed,beta1,beta2,eps,maxEpoch,
        tracking,record,fileExtension);
    }
    else if(famille_algo=="Sperso")
    {
        study = train_Sperso(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,algo,learning_rate,batch_size,2,10000,Sseed,eps,maxEpoch,
        tracking,record,fileExtension);
    }

    return study;

}
