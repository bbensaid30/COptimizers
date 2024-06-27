#include <iostream>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

#include <omp.h>

#include "init.h"
#include "data.h"
#include "training.h"
#include "Straining.h"
#include "test.h"
#include "Stest.h"
#include "tirage.h"
#include "Stirage.h"

int main()
{
    //omp_set_num_threads(omp_get_num_procs());
    omp_set_num_threads(1);
    Eigen::initParallel();

    //Paramètres généraux
    std::string const distribution="uniform";
    std::vector<double> const supParameters={-5,5};
    int const tirageMin=0;
    int const nbTirages=28000;
    std::string const famille_algo="Incremental";
    std::string const algo="RAG_L_ancien";
    double const eps=std::pow(10,-4);
    int const maxEpoch=200000;
    double const epsNeight=5*std::pow(10,-2);

    //Paramètres des méthodes LM
    double mu=10, factor=10, RMin=0.25, RMax=0.75, epsDiag=std::pow(10,-16), Rlim=std::pow(10,-4), factorMin=std::pow(10,-8), power=1.0, alphaChap=1.1, alpha=0.75, pas=0.1;
    double tau=1, beta=2.0, gamma=3.0;
    int const b=1, p=3;

    double learning_rate=0.1;
    double clip=1/learning_rate;
    double seuil=0.01;
    double beta1 = 1-0.9;
    double beta2 = 1-0.999;
    int const batch_size = 1;

    Eigen::MatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyFive";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }

    /* std::map<std::string,double> study;
    std::string fileExtension="polyFive_(100,100)";
    initialisation(nbNeurons,weights,bias,supParameters,distribution,7);
    weights[0](0,0) = 50; bias[0](0) = -50;
    study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,RMin,
    RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,true,true,true,fileExtension);
    std::cout << "Poids final: (" << weights[0](0,0) << ", " << bias[0](0) << ") " << std::endl;
    std::cout << "gradientNorm: " << study["finalGradient"] << std::endl;
    std::cout << "gradientNorm: " << study["finalGradient"].digits() << std::endl;
    std::cout << "iters: " << study["epoch"] << std::endl;
    std::cout << "energy: " << study["prop_entropie"] << std::endl; */

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    std::string setHyperparameters="1";
    test_PolyFive(distribution,supParameters,nbTirages,famille_algo,algo,learning_rate,clip,seuil,beta1,beta2,batch_size,mu,factor,Rlim,RMin,RMax,epsDiag,b,factorMin,
    power,alphaChap,alpha,pas,eps,maxEpoch,epsNeight,true,true,true,setHyperparameters);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "temps: " << time << " s" << std::endl;
}

