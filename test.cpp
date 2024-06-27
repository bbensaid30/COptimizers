#include "test.h"

void test_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor, double const& Rlim, double const& RMin,
double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap, double const& alpha, double const& pas,
double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{

    std::ofstream gradientNormFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"epoch"+".csv").c_str());
    std::ofstream meanForwardFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"meanForward"+".csv").c_str());
    std::ofstream initFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream znFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"zn"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::MatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyTwo";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }
    
    std::vector<int> numero_list(nbTirage);
    std::vector<double> iters_list(nbTirage), iters_MeanForward_list(nbTirage), zn_list(nbTirage);
    std::vector<double> weights_list(nbTirage), bias_list(nbTirage);
    std::vector<double> gradientNorm_list(nbTirage); 
    double normalization = 2/batch_size; 

    std::vector<double> proportions(4,0.0), iters(4,0.0), meanForward(4,0.0), zn(4,0.0);
    int farMin=0, nonConv=0, div=0;

    std::vector<Eigen::VectorXd> points(4);
    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> weights(L);
        std::vector<Eigen::VectorXd> bias(L);
        std::vector<Eigen::MatrixXd> weightsInit(L);
        std::vector<Eigen::VectorXd> biasInit(L);

        std::map<std::string,double> study;
        Eigen::VectorXd currentPoint(2);
        int numeroPoint;
        int i;
    
    #pragma omp for
    for(i=0;i<nbTirage;i++)
    {   
        std::cout << "Tirage: " << i << std::endl;

        initialisation(nbNeurons,weights,bias,supParameters,distribution,i);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (study["finalGradient"]<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = numero_point(currentPoint,points,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl;}
        }
        else
        {
            if(std::abs(study["finalGradient"])>1000 || std::isnan(study["finalGradient"]) || std::isinf(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence: " << i << std::endl;
                numeroPoint = -2;
            }
        }
        numero_list[i] = numeroPoint;
        iters_list[i] = study["epoch"];
        iters_MeanForward_list[i]=study["total_iterLoop"]/(normalization*study["epoch"]);
        zn_list[i]=study["zn"]/(normalization*study["epoch"]);
        gradientNorm_list[i] = study["finalGradient"];
        weights_list[i]=weightsInit[0](0,0); bias_list[i]=biasInit[0](0);
    }
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "temps: " << time << " s" << std::endl;

    for(int i=0; i<nbTirage; i++)
    {
        if(numero_list[i]==-1){farMin++;}
        else if(numero_list[i]==-2){nonConv++;}
        else if(numero_list[i]==-3){div++;}
        else
        {
            proportions[numero_list[i]]++;
            iters[numero_list[i]]+=iters_list[i];
            meanForward[numero_list[i]]+=iters_MeanForward_list[i];
            zn[numero_list[i]]+=zn_list[i];
        }

        if(record)
        {
            gradientNormFlux << numero_list[i] << std::endl;
            gradientNormFlux << gradientNorm_list[i] << std::endl;

            iterFlux << numero_list[i] << std::endl;
            iterFlux << iters_list[i] << std::endl;

            meanForwardFlux << numero_list[i] << std::endl;
            meanForwardFlux << iters_MeanForward_list[i] << std::endl;

            znFlux << numero_list[i] << std::endl;
            znFlux << zn_list[i] << std::endl;

            initFlux << numero_list[i] << std::endl;
            initFlux << weights_list[i] << std::endl;
            initFlux << bias_list[i] << std::endl;
        }
    }

    for(int i=0;i<4;i++)
    {
        if (proportions[i]!=0){iters[i]/=proportions[i]; meanForward[i]/=proportions[i]; zn[i]/=proportions[i];}
        proportions[i]/=double(nbTirage);
    }

    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-2,1): " << meanForward[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (2,-1): " << meanForward[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,-1): " << meanForward[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,1): " << meanForward[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de nonConv: " << double(nonConv)/double(nbTirage) << std::endl;
    std::cout << "Proportion de div: " << double(div)/double(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << double(farMin)/double(nbTirage) << std::endl;

}

void test_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor, double const& Rlim, double const& RMin,
double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap, double const& alpha, double const& pas,
double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{
    std::ofstream gradientNormFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"epoch"+".csv").c_str());
    std::ofstream meanForwardFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"meanForward"+".csv").c_str());
    std::ofstream initFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream znFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"zn"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::MatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyThree";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }
    
    std::vector<int> numero_list(nbTirage);
    std::vector<double> iters_list(nbTirage), iters_MeanForward_list(nbTirage), zn_list(nbTirage);
    std::vector<double> weights_list(nbTirage), bias_list(nbTirage);
    std::vector<double> gradientNorm_list(nbTirage);  

    std::vector<double> proportions(4,0.0), iters(4,0.0), meanForward(4,0.0), zn(4,0.0);
    int farMin=0, nonConv=0, div=0;
    double normalization = 2/batch_size;

    std::vector<Eigen::VectorXd> points(4);
     points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> weights(L);
        std::vector<Eigen::VectorXd> bias(L);
        std::vector<Eigen::MatrixXd> weightsInit(L);
        std::vector<Eigen::VectorXd> biasInit(L);

        std::map<std::string,double> study;
        Eigen::VectorXd currentPoint(2);
        int numeroPoint;
        int i;

    #pragma omp for
    for(i=0;i<nbTirage;i++)
    {   
        std::cout << "Tirage: " << i << std::endl;

        initialisation(nbNeurons,weights,bias,supParameters,distribution,i);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (study["finalGradient"]<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = numero_point(currentPoint,points,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl;}
        }
        else
        {
            if(std::abs(study["finalGradient"])>1000 || std::isnan(study["finalGradient"]) || std::isinf(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence: " << i << std::endl;
                numeroPoint = -2;
            }
        }
        numero_list[i] = numeroPoint;
        iters_list[i] = study["epoch"];
        iters_MeanForward_list[i]=study["total_iterLoop"]/(normalization*study["epoch"]);
        zn_list[i]=study["zn"]/(normalization*study["epoch"]);
        gradientNorm_list[i] = study["finalGradient"];
        weights_list[i]=weightsInit[0](0,0); bias_list[i]=biasInit[0](0);
    }
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "temps: " << time << " s" << std::endl;

    for(int i=0; i<nbTirage; i++)
    {
        if(numero_list[i]==-1){farMin++;}
        else if(numero_list[i]==-2){nonConv++;}
        else if(numero_list[i]==-3){div++;}
        else
        {
            proportions[numero_list[i]]++;
            iters[numero_list[i]]+=iters_list[i];
            meanForward[numero_list[i]]+=iters_MeanForward_list[i];
            zn[numero_list[i]]+=zn_list[i];
        }

        if(record)
        {
            gradientNormFlux << numero_list[i] << std::endl;
            gradientNormFlux << gradientNorm_list[i] << std::endl;

            iterFlux << numero_list[i] << std::endl;
            iterFlux << iters_list[i] << std::endl;

            meanForwardFlux << numero_list[i] << std::endl;
            meanForwardFlux << iters_MeanForward_list[i] << std::endl;

            znFlux << numero_list[i] << std::endl;
            znFlux << zn_list[i] << std::endl;

            initFlux << numero_list[i] << std::endl;
            initFlux << weights_list[i] << std::endl;
            initFlux << bias_list[i] << std::endl;
        }
    }

    for(int i=0;i<4;i++)
    {
        if (proportions[i]!=0){iters[i]/=proportions[i]; meanForward[i]/=proportions[i]; zn[i]/=proportions[i];}
        proportions[i]/=double(nbTirage);
    }

    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-2,1): " << meanForward[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (2,-1): " << meanForward[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,-1): " << meanForward[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,1): " << meanForward[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de nonConv: " << double(nonConv)/double(nbTirage) << std::endl;
    std::cout << "Proportion de div: " << double(div)/double(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << double(farMin)/double(nbTirage) << std::endl;

}

void test_PolyFour(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor, double const& Rlim, double const& RMin,
double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap, double const& alpha, double const& pas,
double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{

    std::ofstream gradientNormFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"epoch"+".csv").c_str());
    std::ofstream meanForwardFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"meanForward"+".csv").c_str());
    std::ofstream initFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream znFlux(("Record/polyFour/"+setHyperparameters+"/"+algo+"_"+"zn"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::MatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyFour";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }
    
    std::vector<int> numero_list(nbTirage);
    std::vector<double> iters_list(nbTirage), iters_MeanForward_list(nbTirage), zn_list(nbTirage);
    std::vector<double> weights_list(nbTirage), bias_list(nbTirage);
    std::vector<double> gradientNorm_list(nbTirage);  

    std::vector<double> proportions(4,0.0), iters(4,0.0), meanForward(4,0.0), zn(4,0.0);
    int farMin=0, nonConv=0, div=0;
    double normalization = 2/batch_size;

    std::vector<Eigen::VectorXd> points(4);
    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2);
    points[0](0)=-2; points[0](1)=1; points[1](0)=2; points[1](1)=-1; points[2](0)=0; points[2](1)=-1; points[3](0)=0; points[3](1)=1;

    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> weights(L);
        std::vector<Eigen::VectorXd> bias(L);
        std::vector<Eigen::MatrixXd> weightsInit(L);
        std::vector<Eigen::VectorXd> biasInit(L);

        std::map<std::string,double> study;
        Eigen::VectorXd currentPoint(2);
        int numeroPoint;
        int i;

    #pragma omp for
    for(i=0;i<nbTirage;i++)
    {   
        std::cout << "Tirage: " << i << std::endl;

        initialisation(nbNeurons,weights,bias,supParameters,distribution,i);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (study["finalGradient"]<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = numero_point(currentPoint,points,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl;}
        }
        else
        {
            if(std::abs(study["finalGradient"])>1000 || std::isnan(study["finalGradient"]) || std::isinf(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence: " << i << std::endl;
                numeroPoint = -2;
            }
        }
        numero_list[i] = numeroPoint;
        iters_list[i] = study["epoch"];
        iters_MeanForward_list[i]=study["total_iterLoop"]/(normalization*study["epoch"]);
        zn_list[i]=study["zn"]/(normalization*study["epoch"]);
        gradientNorm_list[i] = study["finalGradient"];
        weights_list[i]=weightsInit[0](0,0); bias_list[i]=biasInit[0](0);
    }
    }

    for(int i=0; i<nbTirage; i++)
    {
        if(numero_list[i]==-1){farMin++;}
        else if(numero_list[i]==-2){nonConv++;}
        else if(numero_list[i]==-3){div++;}
        else
        {
            proportions[numero_list[i]]++;
            iters[numero_list[i]]+=iters_list[i];
            meanForward[numero_list[i]]+=iters_MeanForward_list[i];
            zn[numero_list[i]]+=zn_list[i];
        }

        if(record)
        {
            gradientNormFlux << numero_list[i] << std::endl;
            gradientNormFlux << gradientNorm_list[i] << std::endl;

            iterFlux << numero_list[i] << std::endl;
            iterFlux << iters_list[i] << std::endl;

            meanForwardFlux << numero_list[i] << std::endl;
            meanForwardFlux << iters_MeanForward_list[i] << std::endl;

            znFlux << numero_list[i] << std::endl;
            znFlux << zn_list[i] << std::endl;

            initFlux << numero_list[i] << std::endl;
            initFlux << weights_list[i] << std::endl;
            initFlux << bias_list[i] << std::endl;
        }
    }

    for(int i=0;i<4;i++)
    {
        if (proportions[i]!=0){iters[i]/=proportions[i]; meanForward[i]/=proportions[i]; zn[i]/=proportions[i];}
        proportions[i]/=double(nbTirage);
    }

    std::cout << "La proportion pour (-2,1): " << proportions[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[0]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-2,1): " << meanForward[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[1]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (2,-1): " << meanForward[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[2]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,-1): " << meanForward[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[3]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,1): " << meanForward[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de nonConv: " << double(nonConv)/double(nbTirage) << std::endl;
    std::cout << "Proportion de div: " << double(div)/double(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un minimum alors que la condition sur le gradient est respectée: " << double(farMin)/double(nbTirage) << std::endl;
}

void test_PolyFive(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2,
int const& batch_size, double& mu, double& factor, double const& Rlim, double const& RMin,
double const& RMax, double const& epsDiag, int const& b, double& factorMin, double const& power, double const& alphaChap, double const& alpha, double const& pas,
double const& eps, int const& maxEpoch, double const& epsNeight,
bool const tracking, bool const track_continuous, bool const record, std::string const setHyperparameters)
{

    std::ofstream gradientNormFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"gradientNorm"+".csv").c_str());
    std::ofstream iterFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"epoch"+".csv").c_str());
    std::ofstream meanForwardFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"meanForward"+".csv").c_str());
    std::ofstream initFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream znFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"zn"+".csv").c_str());
    if(!gradientNormFlux || !iterFlux || !initFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

    Eigen::MatrixXd X(1,2), Y(1,2);
    X(0,0)=0; X(0,1)=1; Y(0,0)=0; Y(0,1)=0;
    int const n0=X.rows(), nL=Y.rows();
    int N=0;
    int const L=1;
    std::vector<int> nbNeurons(L+1);
    std::vector<int> globalIndices(2*L);
    std::vector<std::string> activations(L);

    //Architecture
    int l;
    nbNeurons[0]=n0;
    nbNeurons[1]=nL;
    activations[0]="polyFive";
    for(l=0;l<L;l++)
    {
        N+=nbNeurons[l]*nbNeurons[l+1]; globalIndices[2*l]=N; N+=nbNeurons[l+1]; globalIndices[2*l+1]=N;
    }
    
    std::vector<int> numero_list(nbTirage);
    std::vector<double> iters_list(nbTirage), iters_MeanForward_list(nbTirage), zn_list(nbTirage);
    std::vector<double> weights_list(nbTirage), bias_list(nbTirage);
    std::vector<double> gradientNorm_list(nbTirage);  

    std::vector<double> proportions(15,0.0), iters(15,0.0), meanForward(15,0.0), zn(15,0.0);
    int farMin=0, nonConv=0, div=0;
    double normalization = 2/batch_size;

    std::vector<Eigen::VectorXd> points(15);
    points[0]=Eigen::VectorXd::Zero(2); points[1]=Eigen::VectorXd::Zero(2); points[2]=Eigen::VectorXd::Zero(2); points[3]=Eigen::VectorXd::Zero(2);
    points[4]=Eigen::VectorXd::Zero(2); points[5]=Eigen::VectorXd::Zero(2); points[6]=Eigen::VectorXd::Zero(2); points[7]=Eigen::VectorXd::Zero(2);
    points[8]=Eigen::VectorXd::Zero(2); points[9]=Eigen::VectorXd::Zero(2); points[10]=Eigen::VectorXd::Zero(2); points[11]=Eigen::VectorXd::Zero(2);
    points[12]=Eigen::VectorXd::Zero(2); points[13]=Eigen::VectorXd::Zero(2); points[14]=Eigen::VectorXd::Zero(2);

    //minimums
    points[0](0)=2; points[0](1)=1; points[1](0)=0; points[1](1)=-1; points[2](0)=-2; points[2](1)=3; points[3](0)=0; points[3](1)=3;
    points[4](0)=-4; points[4](1)=3; points[5](0)=4; points[5](1)=-1;
    //saddle points
    points[6](0)=-2; points[6](1)=1; points[7](0)=2; points[7](1)=-1; points[8](0)=0; points[8](1)=1; points[9](0)=4/5; points[9](1)=11/5;
    points[10](0)=-4/5; points[10](1)=3;
    points[11](0)=16/5; points[11](1)=-1; points[12](0)=6/5; points[12](1)=1; points[13](0)=-6/5; points[13](1)=11/5; points[14](0)=0; points[14](1)=11/5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> weights(L);
        std::vector<Eigen::VectorXd> bias(L);
        std::vector<Eigen::MatrixXd> weightsInit(L);
        std::vector<Eigen::VectorXd> biasInit(L);

        std::map<std::string,double> study;
        Eigen::VectorXd currentPoint(2);
        int numeroPoint;
        int i;

    #pragma omp for
    for(i=0;i<nbTirage;i++)
    {   
        std::cout << "Tirage: " << i << std::endl;

        initialisation(nbNeurons,weights,bias,supParameters,distribution,i);
        std::copy(weights.begin(),weights.end(),weightsInit.begin()); std::copy(bias.begin(),bias.end(),biasInit.begin());
        study = train(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);

        if (study["finalGradient"]<eps)
        {
            currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
            numeroPoint = numero_point(currentPoint,points,epsNeight);
            if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl;}
        }
        else
        {
            if(std::abs(study["finalGradient"])>1000 || std::isnan(study["finalGradient"]) || std::isinf(study["finalGradient"]))
            {
                std::cout << "Divergence: " << i << std::endl;
                numeroPoint = -3;
            }
            else
            {
                std::cout << "Non convergence: " << i << std::endl;
                numeroPoint = -2;
            }
        }
        numero_list[i] = numeroPoint;
        iters_list[i] = study["epoch"];
        iters_MeanForward_list[i]=study["total_iterLoop"]/(normalization*study["epoch"]);
        zn_list[i]=study["zn"]/(normalization*study["epoch"]);
        gradientNorm_list[i] = study["finalGradient"];
        weights_list[i]=weightsInit[0](0,0); bias_list[i]=biasInit[0](0);
    }
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "temps: " << time << " s" << std::endl;

    for(int i=0; i<nbTirage; i++)
    {
        if(numero_list[i]==-1){farMin++;}
        else if(numero_list[i]==-2){nonConv++;}
        else if(numero_list[i]==-3){div++;}
        else
        {
            proportions[numero_list[i]]++;
            iters[numero_list[i]]+=iters_list[i];
            meanForward[numero_list[i]]+=iters_MeanForward_list[i];
            zn[numero_list[i]]+=zn_list[i];
        }

        if(record)
        {
            gradientNormFlux << numero_list[i] << std::endl;
            gradientNormFlux << gradientNorm_list[i] << std::endl;

            iterFlux << numero_list[i] << std::endl;
            iterFlux << iters_list[i] << std::endl;

            meanForwardFlux << numero_list[i] << std::endl;
            meanForwardFlux << iters_MeanForward_list[i] << std::endl;

            znFlux << numero_list[i] << std::endl;
            znFlux << zn_list[i] << std::endl;

            initFlux << numero_list[i] << std::endl;
            initFlux << weights_list[i] << std::endl;
            initFlux << bias_list[i] << std::endl;
        }
    }

    for(int i=0;i<15;i++)
    {
        if (proportions[i]!=0){iters[i]/=proportions[i]; meanForward[i]/=proportions[i]; zn[i]/=proportions[i];}
        proportions[i]/=double(nbTirage);
    }

    std::cout << "La proportion pour (2,1): " << proportions[0] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,1): " << iters[0]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (2,1): " << meanForward[0]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,-1): " << proportions[1] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,-1): " << iters[1]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,-1): " << meanForward[1]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-2,3): " << proportions[2] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,3): " << iters[2]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-2,3): " << meanForward[2]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,3): " << proportions[3] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,3): " << iters[3]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,3): " << meanForward[3]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-4,3): " << proportions[4] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-4,3): " << iters[4]<< std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-4,3): " << meanForward[4]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (4,-1): " << proportions[5] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (4,-1): " << iters[5] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (4,-1): " << meanForward[5]<< std::endl;
    std::cout << "-----------------------------------------------------------------------------------------" << std::endl;

    std::cout << "La proportion pour (-2,1): " << proportions[6] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-2,1): " << iters[6] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-2,1): " << meanForward[6]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (2,-1): " << proportions[7] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (2,-1): " << iters[7] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (2,-1): " << meanForward[7]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,1): " << proportions[8] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,1): " << iters[8] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,1): " << meanForward[8]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (4/5,11/5): " << proportions[9] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (4/5,11/5): " << iters[9] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (4/5,11/5): " << meanForward[9]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-4/5,3): " << proportions[10] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-4/5,3): " << iters[10] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-4/5,3): " << meanForward[10]<< std::endl;
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;

    std::cout << "La proportion pour (16/5,-1): " << proportions[11] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (16/5,-1): " << iters[11] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (16/5,-1): " << meanForward[11]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (6/5,1): " << proportions[12] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (6/5,1): " << iters[12] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (6/5,1): " << meanForward[12]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (-6/5,11/5): " << proportions[13] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (-6/5,11/5): " << iters[13] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (-6/5,11/5): " << meanForward[13]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "La proportion pour (0,11/5): " << proportions[14] << std::endl;
    std::cout << "Le nombre moyen d'itérations pour arriver à (0,11/5): " << iters[14] << std::endl;
    std::cout << "Le nombre moyen d'itérations forward pour arriver à (0,11/5): " << meanForward[14]<< std::endl;
    std::cout << "" << std::endl;

    std::cout << "Proportion de nonConv: " << double(nonConv)/double(nbTirage) << std::endl;
    std::cout << "Proportion de div: " << double(div)/double(nbTirage) << std::endl;
    std::cout << "Proportion de fois où on n'est pas assez proche d'un point critique alors que la condition sur le gradient est respectée: " << double(farMin)/double(nbTirage) << std::endl;

}
