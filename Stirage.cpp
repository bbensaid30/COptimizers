#include "Stirage.h"

std::vector<std::map<std::string,Sdouble>> StiragesRegression(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking)
{
    int const PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,Sdouble>> studies(nbTirages*nbSeeds);

    #pragma omp parallel
    {
        std::vector<Eigen::SMatrixXd> weights(L);
        std::vector<Eigen::SVectorXd> bias(L);
        std::vector<Eigen::SMatrixXd> AsTest(L+1);
        AsTest[0]=data[2];
        std::vector<Eigen::SMatrixXd> slopes(L);
        Sdouble costTest;

        #pragma omp for
        for(int i=tirageMin;i<tirageMax;i++)
        {
            for(int j=0; j<nbSeeds; j++)
            {   
                initialisation(nbNeurons,weights,bias,supParameters,generator,i);
                studies[i*nbSeeds+j] = Strain(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
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

std::vector<std::map<std::string,Sdouble>> StiragesClassification(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, int const nbSeeds,
bool const tracking)
{
    int const PTrain=data[0].cols(), PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,Sdouble>> studies(nbTirages*nbSeeds);


    #pragma omp parallel
    {
        std::vector<Eigen::SMatrixXd> weights(L);
        std::vector<Eigen::SVectorXd> bias(L);
        std::vector<Eigen::SMatrixXd> AsTrain(L+1), AsTest(L+1);
        AsTrain[0]=data[0]; AsTest[0]=data[2];
        std::vector<Eigen::SMatrixXd> slopes(L);
        Sdouble costTest;
        Sdouble rateTrain, rateTest;

        #pragma omp for
        for(int i=tirageMin;i<tirageMax;i++)
        {   
            for(int j=0; j<nbSeeds; j++)
            {
                initialisation(nbNeurons,weights,bias,supParameters,generator,i);
                studies[i*nbSeeds+j] = Strain(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
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

void SminsRecordRegression(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTrajs = studies.size();
    int nonConv=0, div=0, numNoise=0;

    std::map<std::string,Sdouble> study;

    for(int i=0; i<nbTrajs; i++)
    {
        study = studies[i];

        if((study["finalGradient"]+std::abs(study["finalGradient"].error)<eps) && !Sstd::isnan(study["finalGradient"]) && !Sstd::isinf(study["finalGradient"]) && !numericalNoise(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"].number << std::endl;
            infosFlux << study["num_seed"].number << std::endl;
            infosFlux << study["iter"].number << std::endl;
            infosFlux << study["time"].number << std::endl;

            infosFlux << study["finalCost"].number << std::endl;
            infosFlux << study["cost_test"].number << std::endl;
            infosFlux << study["prop_entropie"].number << std::endl;
            infosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;

            //------------------------------------------------------

            allinfosFlux << study["num_tirage"].number << std::endl;
            allinfosFlux << study["iter"].number << std::endl;
            allinfosFlux << study["time"].number << std::endl;

            allinfosFlux << study["finalCost"].number << std::endl;
            allinfosFlux << study["cost_test"].number << std::endl;
            allinfosFlux << study["prop_entropie"].number << std::endl;
            allinfosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;

        }
        else
        {
            if(Sstd::abs(study["finalGradient"])>1000 || Sstd::isnan(study["finalGradient"]) || Sstd::isinf(study["finalGradient"]))
            {
                div++;
                nonFlux << -3 << std::endl;
                nonFlux << study["prop_entropie"].number << std::endl;
            }
            else if(numericalNoise(study["finalGradient"]))
            {
                numNoise++;
                nonFlux << -4 << std::endl;
                nonFlux << study["prop_entropie"].number << std::endl;
            }
            else
            {
                //std::cout << study["finalGradient"].number << std::endl;
                nonConv++;
                nonFlux << -2 << std::endl;
                nonFlux << study["prop_entropie"].number << std::endl;

                allinfosFlux << study["num_tirage"].number << std::endl;
                allinfosFlux << study["iter"].number << std::endl;
                allinfosFlux << study["time"].number << std::endl;

                allinfosFlux << study["finalCost"].number << std::endl;
                allinfosFlux << study["cost_test"].number << std::endl;
                allinfosFlux << study["prop_entropie"].number << std::endl;
                allinfosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;
            }
        }

    }

    infosFlux << (Sdouble(nonConv)/Sdouble(nbTrajs)).number << std::endl;
    infosFlux << (Sdouble(div)/Sdouble(nbTrajs)).number << std::endl;
    infosFlux << (Sdouble(numNoise)/Sdouble(nbTrajs)).number << std::endl;

    allinfosFlux << (Sdouble(nonConv)/Sdouble(nbTrajs)).number << std::endl;
    allinfosFlux << (Sdouble(div)/Sdouble(nbTrajs)).number << std::endl;
    allinfosFlux << (Sdouble(numNoise)/Sdouble(nbTrajs)).number << std::endl;

    std::cout << "Proportion de divergence: " << Sdouble(div)/Sdouble(nbTrajs) << std::endl;
    std::cout << "Proportion de non convergence: " << Sdouble(nonConv)/Sdouble(nbTrajs) << std::endl;
    std::cout << "Proportion de numerical noise: " << Sdouble(numNoise)/Sdouble(nbTrajs) << std::endl;

}

void SminsRecordClassification(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTrajs = studies.size();
    int nonConv=0, div=0, numNoise=0;

    std::map<std::string,Sdouble> study;

    for(int i=0; i<nbTrajs; i++)
    {
        study = studies[i];

        if((study["finalGradient"]+std::abs(study["finalGradient"].error)<eps) && !Sstd::isnan(study["finalGradient"]) && !Sstd::isinf(study["finalGradient"]) && !numericalNoise(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"].number << std::endl;
            infosFlux << study["num_seed"].number << std::endl;
            infosFlux << study["iter"].number << std::endl;
            infosFlux << study["time"].number << std::endl;

            infosFlux << study["finalCost"].number << std::endl;
            infosFlux << study["cost_test"].number << std::endl;
            infosFlux << study["prop_entropie"].number << std::endl;
            infosFlux << study["classTrain"].number << std::endl;
            infosFlux << study["classTest"].number << std::endl;
            infosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;

            //---------------------------------------------------

            allinfosFlux << study["num_tirage"].number << std::endl;
            allinfosFlux << study["iter"].number << std::endl;
            allinfosFlux << study["time"].number << std::endl;

            allinfosFlux << study["finalCost"].number << std::endl;
            allinfosFlux << study["cost_test"].number << std::endl;
            allinfosFlux << study["prop_entropie"].number << std::endl;
            allinfosFlux << study["classTrain"].number << std::endl;
            allinfosFlux << study["classTest"].number << std::endl;
            allinfosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;


        }
        else
        {
            if(Sstd::abs(study["finalGradient"])>1000 || Sstd::isnan(study["finalGradient"]) || Sstd::isinf(study["finalGradient"]))
            {
                div++;
                nonFlux << -3 << std::endl;
                nonFlux << study["prop_entropie"].number << std::endl;
            }
            else if(numericalNoise(study["finalGradient"]))
            {
                numNoise++;
                nonFlux << -4 << std::endl;
                nonFlux << study["prop_entropie"].number << std::endl;
            }
            else
            {
                std::cout << study["finalGradient"].number << std::endl;
                nonConv++;
                nonFlux << -2 << std::endl;
                nonFlux << study["prop_entropie"].number << std::endl;

                allinfosFlux << study["num_tirage"].number << std::endl;
                allinfosFlux << study["iter"].number << std::endl;
                allinfosFlux << study["time"].number << std::endl;

                allinfosFlux << study["finalCost"].number << std::endl;
                allinfosFlux << study["cost_test"].number << std::endl;
                allinfosFlux << study["prop_entropie"].number << std::endl;
                allinfosFlux << study["classTrain"].number << std::endl;
                allinfosFlux << study["classTest"].number << std::endl;
                allinfosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;
            }
        }

    }

    infosFlux << (Sdouble(nonConv)/Sdouble(nbTrajs)).number << std::endl;
    infosFlux << (Sdouble(div)/Sdouble(nbTrajs)).number << std::endl;
    infosFlux << (Sdouble(numNoise)/Sdouble(nbTrajs)).number << std::endl;

    allinfosFlux << (Sdouble(nonConv)/Sdouble(nbTrajs)).number << std::endl;
    allinfosFlux << (Sdouble(div)/Sdouble(nbTrajs)).number << std::endl;
    allinfosFlux << (Sdouble(numNoise)/Sdouble(nbTrajs)).number << std::endl;

    std::cout << "Proportion de divergence: " << Sdouble(div)/Sdouble(nbTrajs) << std::endl;
    std::cout << "Proportion de non convergence: " << Sdouble(nonConv)/Sdouble(nbTrajs) << std::endl;
    std::cout << "Proportion de numerical noise: " << Sdouble(numNoise)/Sdouble(nbTrajs) << std::endl;

}

std::string SinformationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eta, int const& batch_size, Sdouble const& eps, int const& maxIter, std::string const fileExtension)
{
    std::ostringstream epsStream;
    epsStream << eps.number;
    std::string epsString = epsStream.str();
    std::ostringstream etaStream;
    etaStream << eta.number;
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
    maxIterStream << maxIter;
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


    return algo+"("+fileExtension+")"+archi+"(eta="+etaString+", b="+batchString+", eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+", maxIter="+maxIterString+")"+ gen +".csv";
}