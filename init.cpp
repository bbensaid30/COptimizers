#include "init.h"

void simple(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed)
{
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for(l=0;l<L;l++)
    {
        weights[l] = convertToShaman(Eigen::Rand::balanced<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen));
        bias[l] = convertToShaman(Eigen::Rand::balanced<Eigen::MatrixXd>(nbNeurons[l+1],1,gen));
    }
}

void uniform(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, double const& a, double const& b, unsigned const seed)
{

    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        weights[l] = convertToShaman((b-a)*Eigen::Rand::uniformReal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen).array()+a);
        bias[l] = convertToShaman((b-a)*Eigen::Rand::uniformReal<Eigen::MatrixXd>(nbNeurons[l+1],1,gen).array()+a);
    }
}

void normal(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, double const& mu, double const& sigma, unsigned const seed)
{
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        weights[l] = convertToShaman(Eigen::Rand::normal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen,mu,sigma));
        bias[l] = convertToShaman(Eigen::Rand::normal<Eigen::MatrixXd>(nbNeurons[l+1],1,gen,mu,sigma));
    }
}



void Xavier(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed)
{
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        weights[l] = Sstd::sqrt(1/Sdouble(nbNeurons[l]))*convertToShaman(Eigen::Rand::normal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen,0,1));
        bias[l] = Eigen::SVectorXd::Zero(nbNeurons[l+1]);
    }
}
void He(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed)
{
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        weights[l] = Sstd::sqrt(2/Sdouble(nbNeurons[l]))*convertToShaman(Eigen::Rand::normal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen,0,1));
        bias[l] = Eigen::SVectorXd::Zero(nbNeurons[l+1]);
    }
}
void Kaiming(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed)
{
    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;

    for (l=0;l<L;l++)
    {
        weights[l] = Sstd::sqrt(2/Sdouble(nbNeurons[l]+nbNeurons[l+1]))*convertToShaman(Eigen::Rand::normal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen,0,1));
        bias[l] = Eigen::SVectorXd::Zero(nbNeurons[l+1]);
    }
}
void Bergio(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, unsigned const seed)
{

    Eigen::Rand::Vmt19937_64 gen{ seed };

    int const L = nbNeurons.size()-1;
    int l;
    double a,b;

    for (l=0;l<L;l++)
    {
        b = std::sqrt(6/(double)(nbNeurons[l]+nbNeurons[l+1])); a=-b;
        weights[l] = convertToShaman((b-a)*Eigen::Rand::uniformReal<Eigen::MatrixXd>(nbNeurons[l+1],nbNeurons[l],gen).array()+a);
        bias[l] = Eigen::SVectorXd::Zero(nbNeurons[l+1]);
    }
}


void initialisation(std::vector<int> const& nbNeurons, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::vector<double> const& supParameters,
std::string const& generator, unsigned const seed)
{
    if (generator=="uniform")
    {
        uniform(nbNeurons,weights,bias,supParameters[0],supParameters[1], seed);
    }
    else if (generator=="normal")
    {
        normal(nbNeurons,weights,bias,supParameters[0],supParameters[1], seed);
    }
    else if (generator=="Xavier")
    {
        Xavier(nbNeurons,weights,bias,seed);
    }
    else if (generator=="He")
    {
        He(nbNeurons,weights,bias,seed);
    }
    else if (generator=="Kaiming")
    {
        Kaiming(nbNeurons,weights,bias,seed);
    }
    else if (generator=="Bergio")
    {
        Bergio(nbNeurons,weights,bias,seed);
    }
    else
    {
        simple(nbNeurons,weights,bias,seed);
    }
}
