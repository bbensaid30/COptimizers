#include "Stest.h"

void Stest_PolyTwo(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& beta1, double const& beta2,
int const& batch_size, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const record, std::string const setHyperparameters)
{
    std::ofstream iterFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"epoch"+".csv").c_str());
    std::ofstream initFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream iterForwardFlux(("Record/polyTwo/"+setHyperparameters+"/"+algo+"_"+"iter_MeanForward"+".csv").c_str());
    if(!iterFlux || !initFlux || !iterForwardFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

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

    int const nbSeeds = 100;
    std::vector<Eigen::VectorXd> points(4);
    std::vector<std::vector<double>> proportions_list(nbTirage), iters_list(nbTirage), iters_MeanForward_list(nbTirage);
    std::vector<double> weights_list(nbTirage), bias_list(nbTirage);
    std::vector<double> farMin_list(nbTirage), nonMin_list(nbTirage);  

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
        std::vector<double> proportions(4,0.0), iters(4,0.0), distances(4,0.0), iters_MeanForward(4,0.0);
        int numeroPoint;
        double farMin, nonMin;

        int i,j,k;

    #pragma omp for
    for(i=0;i<nbTirage;i++)
    {
        initialisation(nbNeurons,weightsInit,biasInit,supParameters,distribution,i);
        std::copy(weightsInit.begin(),weightsInit.end(),weights.begin()); std::copy(biasInit.begin(),biasInit.end(),bias.begin());
        nonMin=0; farMin=0;
        std::fill(proportions.begin(), proportions.end(), 0); std::fill(iters.begin(), iters.end(), 0); std::fill(iters_MeanForward.begin(), iters_MeanForward.end(), 0);
        for(j=0; j<nbSeeds; j++)
        {
            std::copy(weightsInit.begin(),weightsInit.end(),weights.begin()); std::copy(biasInit.begin(),biasInit.end(),bias.begin());
            study = Strain(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,learning_rate,beta1,beta2,
            batch_size,j,false);

            if (study["finalGradient"]<eps)
            {
                currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
                numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
                if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl; farMin++;}
                else{iters[numeroPoint]+=study["epoch"]; iters_MeanForward[numeroPoint]+=study["total_iterLoop"]/study["epoch"];}
            }
            else
            {
                nonMin++;
                //std::cout << "On n'est pas tombé sur un minimum" << std::endl;
            }

        }
        std::cout << "tirage: " << i << std::endl;
        for(k=0;k<4;k++)
        {
            if(proportions[k]!=0){iters[k]/=proportions[k]; iters_MeanForward[k]/=proportions[k];}
            proportions[k]/=double(nbSeeds);
        }
        farMin/=double(nbSeeds); nonMin/=double(nbSeeds);

        proportions_list[i] = proportions; farMin_list[i] = farMin; nonMin_list[i] = nonMin; 
        iters_list[i]=iters; iters_MeanForward_list[i]=iters_MeanForward;
        weights_list[i]=weightsInit[0](0,0); bias_list[i]=biasInit[0](0);
    }
    }

    if(record)
    {
        for(int i=0; i<nbTirage; i++)
        {
            initFlux << weights_list[i] << std::endl;
            initFlux << bias_list[i] << std::endl;
            initFlux << proportions_list[i][0] << std::endl;
            initFlux << proportions_list[i][1] << std::endl;
            initFlux << proportions_list[i][2] << std::endl;
            initFlux << proportions_list[i][3] << std::endl;
            initFlux << farMin_list[i] << std::endl;
            initFlux << nonMin_list[i] << std::endl;

            iterFlux << weights_list[i] << std::endl;
            iterFlux << bias_list[i] << std::endl;
            iterFlux << iters_list[i][0] << std::endl;
            iterFlux << iters_list[i][1] << std::endl;
            iterFlux << iters_list[i][2] << std::endl;
            iterFlux << iters_list[i][3] << std::endl;

            iterForwardFlux << weights_list[i] << std::endl;
            iterForwardFlux << bias_list[i] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][0] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][1] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][2] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][3] << std::endl;
        }
    }

}

void Stest_PolyThree(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& beta1, double const& beta2,
int const& batch_size, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const record, std::string const setHyperparameters)
{
    std::ofstream iterFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"epoch"+".csv").c_str());
    std::ofstream initFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream iterForwardFlux(("Record/polyThree/"+setHyperparameters+"/"+algo+"_"+"iter_MeanForward"+".csv").c_str());
    if(!iterFlux || !initFlux || !iterForwardFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

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

    int const nbSeeds = 100;
    std::vector<Eigen::VectorXd> points(4);
    std::vector<std::vector<double>> proportions_list(nbTirage), iters_list(nbTirage), iters_MeanForward_list(nbTirage);
    std::vector<double> weights_list(nbTirage), bias_list(nbTirage);
    std::vector<double> farMin_list(nbTirage), nonMin_list(nbTirage);  

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
        std::vector<double> proportions(4,0.0), iters(4,0.0), distances(4,0.0), iters_MeanForward(4,0.0);
        int numeroPoint;
        double farMin, nonMin;

        int i,j,k;

    #pragma omp for
    for(i=0;i<nbTirage;i++)
    {
        initialisation(nbNeurons,weightsInit,biasInit,supParameters,distribution,i);
        std::copy(weightsInit.begin(),weightsInit.end(),weights.begin()); std::copy(biasInit.begin(),biasInit.end(),bias.begin());
        nonMin=0; farMin=0;
        std::fill(proportions.begin(), proportions.end(), 0); std::fill(iters.begin(), iters.end(), 0); std::fill(iters_MeanForward.begin(), iters_MeanForward.end(), 0);
        for(j=0; j<nbSeeds; j++)
        {
            std::copy(weightsInit.begin(),weightsInit.end(),weights.begin()); std::copy(biasInit.begin(),biasInit.end(),bias.begin());
            study = Strain(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,learning_rate,beta1,beta2,
            batch_size,j,false);

            if (study["finalGradient"]<eps)
            {
                currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
                numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
                if(numeroPoint<0){std::cout << "On n'est pas assez proche du minimum même si la condition sur le gradient est respectée" << std::endl; farMin++;}
                else{iters[numeroPoint]+=study["epoch"]; iters_MeanForward[numeroPoint]+=study["total_iterLoop"]/study["epoch"];}
            }
            else
            {
                nonMin++;
                //std::cout << "On n'est pas tombé sur un minimum" << std::endl;
            }

        }
        std::cout << "tirage: " << i << std::endl;
        for(k=0;k<4;k++)
        {
            if(proportions[k]!=0){iters[k]/=proportions[k]; iters_MeanForward[k]/=proportions[k];}
            proportions[k]/=double(nbSeeds);
        }
        farMin/=double(nbSeeds); nonMin/=double(nbSeeds);

        proportions_list[i] = proportions; farMin_list[i] = farMin; nonMin_list[i] = nonMin; 
        iters_list[i]=iters; iters_MeanForward_list[i]=iters_MeanForward; 
        weights_list[i]=weightsInit[0](0,0); bias_list[i]=biasInit[0](0);
    }
    }

    if(record)
    {
        for(int i=0; i<nbTirage; i++)
        {
            initFlux << weights_list[i] << std::endl;
            initFlux << bias_list[i] << std::endl;
            initFlux << proportions_list[i][0] << std::endl;
            initFlux << proportions_list[i][1] << std::endl;
            initFlux << proportions_list[i][2] << std::endl;
            initFlux << proportions_list[i][3] << std::endl;
            initFlux << farMin_list[i] << std::endl;
            initFlux << nonMin_list[i] << std::endl;

            iterFlux << weights_list[i] << std::endl;
            iterFlux << bias_list[i] << std::endl;
            iterFlux << iters_list[i][0] << std::endl;
            iterFlux << iters_list[i][1] << std::endl;
            iterFlux << iters_list[i][2] << std::endl;
            iterFlux << iters_list[i][3] << std::endl;

            iterForwardFlux << weights_list[i] << std::endl;
            iterForwardFlux << bias_list[i] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][0] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][1] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][2] << std::endl;
            iterForwardFlux << iters_MeanForward_list[i][3] << std::endl;
        }
    }

}

void Stest_PolyFive(std::string const& distribution, std::vector<double> const& supParameters, int const& nbTirage, std::string const& famille_algo, std::string const& algo,
double const& learning_rate, double const& beta1, double const& beta2,
int const& batch_size, double const& eps, int const& maxEpoch, double const& epsNeight,
bool const record, std::string const setHyperparameters)
{
    std::ofstream iterFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"epoch"+".csv").c_str());
    std::ofstream initFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"init"+".csv").c_str());
    std::ofstream iterForwardFlux(("Record/polyFive/"+setHyperparameters+"/"+algo+"_"+"iter_MeanForward"+".csv").c_str());
    if(!iterFlux || !initFlux || !iterForwardFlux){std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl;}

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

    int const nbSeeds = 100;
    std::vector<Eigen::VectorXd> points(15);
    std::vector<std::vector<double>> proportions_list(nbTirage), iters_list(nbTirage), iters_MeanForward_list(nbTirage);
    std::vector<double> weights_list(nbTirage), bias_list(nbTirage);
    std::vector<double> farMin_list(nbTirage), nonMin_list(nbTirage);  

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
    
    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> weights(L);
        std::vector<Eigen::VectorXd> bias(L);
        std::vector<Eigen::MatrixXd> weightsInit(L);
        std::vector<Eigen::VectorXd> biasInit(L);

        std::map<std::string,double> study;
        Eigen::VectorXd currentPoint(2);
        std::vector<double> proportions(15,0.0), iters(15,0.0), distances(15,0.0), iters_MeanForward(15,0.0);
        int numeroPoint;
        double farMin, nonMin;

        int i,j,k;

    #pragma omp for
    for(i=0;i<nbTirage;i++)
    {
        initialisation(nbNeurons,weightsInit,biasInit,supParameters,distribution,i);
        std::copy(weightsInit.begin(),weightsInit.end(),weights.begin()); std::copy(biasInit.begin(),biasInit.end(),bias.begin());
        nonMin=0; farMin=0;
        std::fill(proportions.begin(), proportions.end(), 0); std::fill(iters.begin(), iters.end(), 0); std::fill(iters_MeanForward.begin(), iters_MeanForward.end(), 0);
        for(j=0; j<nbSeeds; j++)
        {
            std::copy(weightsInit.begin(),weightsInit.end(),weights.begin()); std::copy(biasInit.begin(),biasInit.end(),bias.begin());
            study = Strain(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,"norme2",famille_algo,algo,eps,maxEpoch,learning_rate,beta1,beta2,
            batch_size,j,false);

            if (study["finalGradient"]<eps)
            {
                currentPoint(0)=weights[0](0,0); currentPoint(1)=bias[0](0);
                numeroPoint = proportion(currentPoint,points,proportions,distances,epsNeight);
                if(numeroPoint<0){std::cout << "On n'est pas assez proche du point critique même si la condition sur le gradient est respectée" << std::endl; farMin++;}
                else{iters[numeroPoint]+=study["epoch"]; iters_MeanForward[numeroPoint]+=study["total_iterLoop"]/study["epoch"];}
            }
            else
            {
                nonMin++;
                //std::cout << "On n'est pas tombé sur un minimum" << std::endl;
            }

        }
        std::cout << "tirage: " << i << std::endl;
        for(k=0;k<15;k++)
        {
            if(proportions[k]!=0){iters[k]/=proportions[k]; iters_MeanForward[k]/=proportions[k];}
            proportions[k]/=double(nbSeeds);
        }
        farMin/=double(nbSeeds); nonMin/=double(nbSeeds);

        proportions_list[i] = proportions; farMin_list[i] = farMin; nonMin_list[i] = nonMin; 
        iters_list[i]=iters; iters_MeanForward_list[i]=iters_MeanForward; 
        weights_list[i]=weightsInit[0](0,0); bias_list[i]=biasInit[0](0);
    }
    }

    if(record)
    {
        for(int i=0; i<nbTirage; i++)
        {
            initFlux << weights_list[i] << std::endl;
            initFlux << bias_list[i] << std::endl;
            for(int k=0; k<15; k++)
            {
                initFlux << proportions_list[i][k] << std::endl;
            }
            initFlux << farMin_list[i] << std::endl;
            initFlux << nonMin_list[i] << std::endl;

            iterFlux << weights_list[i] << std::endl;
            iterFlux << bias_list[i] << std::endl;

            iterForwardFlux << weights_list[i] << std::endl;
            iterForwardFlux << bias_list[i] << std::endl;
            
            for(int k=0; k<15; k++)
            {
                iterFlux << iters_list[i][k] << std::endl;
                iterForwardFlux << iters_MeanForward_list[i][k] << std::endl;
            }
        }
    }

}