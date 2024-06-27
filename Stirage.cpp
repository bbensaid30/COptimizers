#include "Stirage.h"

std::vector<std::map<std::string,double>> StiragesRegression(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking)
{
    int const PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,double>> studies(nbTirages*nbSeeds);

    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> weights(L);
        std::vector<Eigen::VectorXd> bias(L);
        std::vector<Eigen::MatrixXd> AsTest(L+1);
        AsTest[0]=data[2];
        std::vector<Eigen::MatrixXd> slopes(L);
        double costTest;

        #pragma omp for
        for(int i=tirageMin;i<tirageMax;i++)
        {
            for(int j=0; j<nbSeeds; j++)
            {   
                initialisation(nbNeurons,weights,bias,supParameters,generator,i);
                studies[i*nbSeeds+j] = Strain(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxEpoch,
                learning_rate,beta1,beta2,batch_size,j);

                fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
                costTest = risk(data[3],PTest,AsTest[L],type_perte);
                studies[i*nbSeeds+j]["cost_test"] = costTest;
                studies[i*nbSeeds+j]["num_tirage"] = i;
                studies[i*nbSeeds+j]["num_seed"] = j;
            }
            std::cout << "On est au tirage: " << i << std::endl;
        }
    }

    return studies;

}

std::vector<std::map<std::string,double>> StiragesClassification(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking)
{
    int const PTrain=data[0].cols(), PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,double>> studies(nbTirages*nbSeeds);


    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> weights(L);
        std::vector<Eigen::VectorXd> bias(L);
        std::vector<Eigen::MatrixXd> AsTrain(L+1), AsTest(L+1);
        AsTrain[0]=data[0]; AsTest[0]=data[2];
        std::vector<Eigen::MatrixXd> slopes(L);
        double costTest;
        double rateTrain, rateTest;

        #pragma omp for
        for(int i=tirageMin;i<tirageMax;i++)
        {   
            for(int j=0; j<nbSeeds; j++)
            {
                initialisation(nbNeurons,weights,bias,supParameters,generator,i);
                studies[i*nbSeeds+j] = Strain(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxEpoch,
                learning_rate,beta1,beta2,batch_size,j);

                fforward(L,PTrain,nbNeurons,activations,weights,bias,AsTrain,slopes);

                fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
                costTest = risk(data[3],PTest,AsTest[L],type_perte);
                studies[i*nbSeeds+j]["cost_test"] = costTest;
                studies[i*nbSeeds+j]["num_tirage"] = i;
                studies[i*nbSeeds+j]["num_seed"] = j;

                classificationRate(data[1],data[3],AsTrain[L],AsTest[L],PTrain,PTest,rateTrain,rateTest);

                studies[i*nbSeeds+j]["classTrain"] = rateTrain; studies[i*nbSeeds+j]["classTest"] = rateTest;
            }
            std::cout << "On est au tirage: " << i << std::endl;
        }
    }

    return studies;

}

void SminsRecordRegression(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTrajs = studies.size();
    int nonConv=0, div=0;

    std::map<std::string,double> study;

    for(int i=0; i<nbTrajs; i++)
    {
        study = studies[i];

        if((study["finalGradient"]<eps) && !std::isnan(study["finalGradient"]) && !std::isinf(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"] << std::endl;
            infosFlux << study["num_seed"] << std::endl;
            infosFlux << study["epoch"] << std::endl;
            infosFlux << study["time"] << std::endl;

            infosFlux << study["finalCost"] << std::endl;
            infosFlux << study["cost_test"] << std::endl;
            infosFlux << study["prop_entropie"] << std::endl;
            infosFlux << study["total_iterLoop"]/study["epoch"] << std::endl;

            //------------------------------------------------------

            allinfosFlux << study["finalGradient"] << std::endl;
            
        }
        else
        {
            if(std::abs(study["finalGradient"])>1000 || std::isnan(study["finalGradient"]) || std::isinf(study["finalGradient"]))
            {
                div++;
                nonFlux << -3 << std::endl;
                nonFlux << study["prop_entropie"] << std::endl;
            }
            else
            {
                //std::cout << study["finalGradient"] << std::endl;
                nonConv++;
                nonFlux << -2 << std::endl;
                nonFlux << study["prop_entropie"] << std::endl;

                allinfosFlux << study["finalGradient"] << std::endl;
            }
        }

    }

    infosFlux << (double(nonConv)/double(nbTrajs)) << std::endl;
    infosFlux << (double(div)/double(nbTrajs)) << std::endl;

    allinfosFlux << (double(nonConv)/double(nbTrajs)) << std::endl;
    allinfosFlux << (double(div)/double(nbTrajs)) << std::endl;

    std::cout << "Proportion de divergence: " << double(div)/double(nbTrajs) << std::endl;
    std::cout << "Proportion de non convergence: " << double(nonConv)/double(nbTrajs) << std::endl;

}

void SminsRecordClassification(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTrajs = studies.size();
    int nonConv=0, div=0;

    std::map<std::string,double> study;

    for(int i=0; i<nbTrajs; i++)
    {
        study = studies[i];

        if((study["finalGradient"]<eps) && !std::isnan(study["finalGradient"]) && !std::isinf(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"] << std::endl;
            infosFlux << study["num_seed"] << std::endl;
            infosFlux << study["epoch"] << std::endl;
            infosFlux << study["time"] << std::endl;

            infosFlux << study["finalCost"] << std::endl;
            infosFlux << study["cost_test"] << std::endl;
            infosFlux << study["prop_entropie"] << std::endl;
            infosFlux << study["classTrain"] << std::endl;
            infosFlux << study["classTest"] << std::endl;
            infosFlux << study["total_iterLoop"]/study["epoch"] << std::endl;

            //---------------------------------------------------

            allinfosFlux << study["finalGradient"] << std::endl;


        }
        else
        {
            if(std::abs(study["finalGradient"])>1000 || std::isnan(study["finalGradient"]) || std::isinf(study["finalGradient"]))
            {
                div++;
                nonFlux << -3 << std::endl;
                nonFlux << study["prop_entropie"] << std::endl;
            }
            else
            {
                std::cout << study["finalGradient"] << std::endl;
                nonConv++;
                nonFlux << -2 << std::endl;
                nonFlux << study["prop_entropie"] << std::endl;

                allinfosFlux << study["finalGradient"] << std::endl;
            }
        }

    }

    infosFlux << (double(nonConv)/double(nbTrajs)) << std::endl;
    infosFlux << (double(div)/double(nbTrajs)) << std::endl;

    allinfosFlux << (double(nonConv)/double(nbTrajs)) << std::endl;
    allinfosFlux << (double(div)/double(nbTrajs)) << std::endl;

    std::cout << "Proportion de divergence: " << double(div)/double(nbTrajs) << std::endl;
    std::cout << "Proportion de non convergence: " << double(nonConv)/double(nbTrajs) << std::endl;

}

std::string SinformationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eta, int const& batch_size, double const& eps, int const& maxEpoch, std::string const fileExtension)
{
    std::ostringstream epsStream;
    epsStream << eps;
    std::string epsString = epsStream.str();
    std::ostringstream etaStream;
    etaStream << eta;
    std::string etaString = etaStream.str();
    std::ostringstream batchStream;
    batchStream << batch_size;
    std::string batchString = batchStream.str();
    std::ostringstream PTrainStream;
    PTrainStream << PTrain;
    std::string PTrainString = PTrainStream.str();
    std::ostringstream PTestStream;
    PTestStream << PTest;
    std::string PTestString = PTestStream.str();
    std::ostringstream tirageMinStream;
    tirageMinStream << tirageMin;
    std::string tirageMinString = tirageMinStream.str();
    std::ostringstream nbTiragesStream;
    nbTiragesStream << nbTirages;
    std::string nbTiragesString = nbTiragesStream.str();
    std::ostringstream maxIterStream;
    maxIterStream << maxEpoch;
    std::string maxIterString = maxIterStream.str();

    std::string archi = "";
    for(int l=0; l<L; l++)
    {
        archi += std::to_string(nbNeurons[l+1]);
        archi+="("; archi += activations[l]; archi += ")";
        archi+="-";
    }

    int tailleParameters = supParameters.size();
    std::string gen = generator; gen+="(";
    if(tailleParameters>0)
    {
        for(int s=0; s<tailleParameters; s++)
        {
            gen += std::to_string(supParameters[s]); gen+=",";
        }
    }
    gen+=")";


    return algo+"("+fileExtension+")"+archi+"(eta="+etaString+", b="+batchString+", eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+", maxEpoch="+maxIterString+")"+ gen +".csv";
}