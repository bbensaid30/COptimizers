#include "tirage.h"

std::vector<std::map<std::string,double>> tiragesRegression(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2, int const & batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
bool const tracking)
{
    int const PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,double>> studies(nbTirages);


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
            initialisation(nbNeurons,weights,bias,supParameters,generator,i);
            studies[i] = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxEpoch,
            learning_rate,clip,seuil,beta1,beta2,batch_size,
            mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking);

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);
            studies[i]["cost_test"] = costTest;
            studies[i]["num_tirage"] = i;

            /* std::cout << "On est au tirage: " << i << std::endl;
            std::cout << "iters: " << studies[i]["epoch"] << std::endl;
            std::cout << "costTrain: " << studies[i]["finalCost"] << std::endl;
            std::cout << "costTest: " << studies[i]["cost_test"] << std::endl;
            std::cout << "Numéro Thread: " << omp_get_thread_num() << std::endl; */
        }
    }

    return studies;

}

std::vector<std::map<std::string,double>> tiragesClassification(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2, int const & batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
bool const tracking)
{
    int const PTrain=data[0].cols(), PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,double>> studies(nbTirages);


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
            initialisation(nbNeurons,weights,bias,supParameters,generator,i);
            studies[i] = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxEpoch,
            learning_rate,clip,seuil,beta1,beta2,batch_size,
            mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking);

            fforward(L,PTrain,nbNeurons,activations,weights,bias,AsTrain,slopes);

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);
            studies[i]["cost_test"] = costTest;
            studies[i]["num_tirage"] = i;

            classificationRate(data[1],data[3],AsTrain[L],AsTest[L],PTrain,PTest,rateTrain,rateTest);

            studies[i]["classTrain"] = rateTrain; studies[i]["classTest"] = rateTest;

            std::cout << "On est au tirage: " << i << std::endl;
            std::cout << "Numéro Thread: " << omp_get_thread_num() << std::endl;
        }
    }

    return studies;

}

void minsRecordRegression(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTirages = studies.size();
    int nonConv=0, div=0;

    std::map<std::string,double> study;

    for(int i=0; i<nbTirages; i++)
    {
        study = studies[i];

        if((study["finalGradient"]<eps) && !std::isnan(study["finalGradient"]) && !std::isinf(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"] << std::endl;
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
                std::cout << study["finalGradient"] << std::endl;
                nonConv++;
                nonFlux << -2 << std::endl;
                nonFlux << study["prop_entropie"] << std::endl;

                allinfosFlux << study["finalGradient"] << std::endl;
            }
        }

    }

    infosFlux << (double(nonConv)/double(nbTirages)) << std::endl;
    infosFlux << (double(div)/double(nbTirages)) << std::endl;

    allinfosFlux << (double(nonConv)/double(nbTirages)) << std::endl;
    allinfosFlux << (double(div)/double(nbTirages)) << std::endl;

    std::cout << "Proportion de divergence: " << double(div)/double(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << double(nonConv)/double(nbTirages) << std::endl;

}

void minsRecordClassification(std::vector<std::map<std::string,double>> studies, std::string const& folder, std::string const& fileEnd, double const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTirages = studies.size();
    int nonConv=0, div=0;

    std::map<std::string,double> study;

    for(int i=0; i<nbTirages; i++)
    {
        study = studies[i];

        if((study["finalGradient"]<eps) && !std::isnan(study["finalGradient"]) && !std::isinf(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"] << std::endl;
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
                //std::cout << study["finalGradient"] << std::endl;
                nonConv++;
                nonFlux << -2 << std::endl;
                nonFlux << study["prop_entropie"] << std::endl;

                allinfosFlux << study["finalGradient"] << std::endl;
            }
        }

    }

    infosFlux << (double(nonConv)/double(nbTirages)) << std::endl;
    infosFlux << (double(div)/double(nbTirages)) << std::endl;

    allinfosFlux << (double(nonConv)/double(nbTirages)) << std::endl;
    allinfosFlux << (double(div)/double(nbTirages)) << std::endl;

    std::cout << "Proportion de divergence: " << double(div)/double(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << double(nonConv)/double(nbTirages) << std::endl;

}

void predictionsRecord(std::vector<Eigen::MatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, double const& eps, int const& maxEpoch,
double const& learning_rate, double const& clip, double const& seuil, double const& beta1, double const& beta2, int const & batch_size,
double& mu, double& factor, double const& RMin, double const& RMax, int const& b, double const& alpha,
double const& pas, double const& Rlim, double& factorMin, double const& power, double const& alphaChap, double const& epsDiag,
std::string const& folder, std::string const fileExtension, bool const tracking, bool const track_continuous)
{

    int const PTrain = data[0].cols();
    int const PTest = data[2].cols();

    std::string const fileEnd = informationFile(PTrain,PTest,L,nbNeurons,activations,type_perte,algo,supParameters,generator,tirageMin,nbTirages,learning_rate,eps,batch_size, maxEpoch);

    std::ofstream costFlux(("Record/"+folder+"/cost_"+fileEnd).c_str());
    std::ofstream costTestFlux(("Record/"+folder+"/costTest_"+fileEnd).c_str());

    std::ofstream inputsFlux(("Record/"+folder+"/inputs_"+fileEnd).c_str());
    std::ofstream bestTrainFlux(("Record/"+folder+"/bestTrain_"+fileEnd).c_str());
    std::ofstream bestTestFlux(("Record/"+folder+"/bestTest_"+fileEnd).c_str());
    std::ofstream moyTrainFlux(("Record/"+folder+"/moyTrain_"+fileEnd).c_str());
    std::ofstream moyTestFlux(("Record/"+folder+"/moyTest_"+fileEnd).c_str());

    std::ofstream trackingFlux(("Record/"+folder+"/tracking_"+fileEnd).c_str());
    std::ofstream trackContinuousFlux(("Record/"+folder+"/track_continuous_"+fileEnd).c_str());


    if(!costFlux || !costTestFlux || !inputsFlux || !bestTrainFlux || !bestTestFlux || !moyTrainFlux || !moyTestFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    unsigned seed;
    std::vector<Eigen::MatrixXd> weights(L);
    std::vector<Eigen::VectorXd> bias(L);
    int const N = globalIndices[2*L-1];

    std::vector<Eigen::MatrixXd> AsTrain(L+1); AsTrain[0]=data[0];
    std::vector<Eigen::MatrixXd> AsTest(L+1); AsTest[0]=data[2];
    std::vector<Eigen::MatrixXd> slopes(L);

    Eigen::MatrixXd bestPredictionsTrain(nbNeurons[L],PTrain), bestPredictionsTest(nbNeurons[L],PTest);
    Eigen::MatrixXd moyPredictionsTrain(nbNeurons[L],PTrain), moyPredictionsTest(nbNeurons[L],PTest);
    moyPredictionsTrain.setZero(); moyPredictionsTest.setZero();

    double costMin=10000, costTest;

    std::map<std::string,double> study;
    int nonConv=0, div=0;

    int const tirageMax=tirageMin+nbTirages;
    int minAttain=0, nMin;

    for(int i=tirageMin; i<tirageMax; i++)
    {
        if(i!=0 && i%100==0)
        {
            std::cout << "On est au tirage" << i << std::endl;
        }

        seed=i; initialisation(nbNeurons,weights,bias,supParameters,generator,seed);
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxEpoch,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);
        if(study["finalGradient"]<eps && !std::isnan(study["finalCost"]) && !std::isinf(study["finalCost"]))
        {
            fforward(L,PTrain,nbNeurons,activations,weights,bias,AsTrain,slopes);

            costFlux << i << std::endl;
            costFlux << study["epoch"] << std::endl;
            costFlux << study["finalCost"] << std::endl;

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);

            costTestFlux << i << std::endl;
            costTestFlux << study["epoch"] << std::endl;
            costTestFlux << costTest << std::endl;

            if(study["finalCost"] < costMin)
            {
                costMin = study["finalCost"];

                bestPredictionsTrain = AsTrain[L];
                bestPredictionsTest = AsTest[L];
            }
            moyPredictionsTrain += AsTrain[L];
            moyPredictionsTest += AsTest[L];

            minAttain++; nMin=0;

        }
        else
        {
            if(std::abs(study["finalGradient"])>1 || std::isnan(study["finalGradient"]))
            {
                div++; nMin=-3;
            }
            else
            {
                nonConv++; nMin=-2;
            }
        }

        if(tracking)
        {
            trackingFlux << nMin << std::endl;
            trackingFlux << study["epoch"] << std::endl;
            trackingFlux << study["prop_entropie"] << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << nMin << std::endl;
            trackContinuousFlux << study["epoch"] << std::endl;
            trackContinuousFlux << study["continuous_entropie"] << std::endl;
        }
    }

    std::cout << "Proportion de divergence: " << double(div)/double(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << double(nonConv)/double(nbTirages) << std::endl;

    bestTrainFlux << bestPredictionsTrain << std::endl;
    bestTestFlux << bestPredictionsTest << std::endl;
    moyTrainFlux << moyPredictionsTrain/double(minAttain) << std::endl;
    moyTestFlux << moyPredictionsTest/double(minAttain) << std::endl;

}


std::string informationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
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

void classificationRate(Eigen::MatrixXd const& YTrain, Eigen::MatrixXd const& YTest, Eigen::MatrixXd const& outputTrain, Eigen::MatrixXd const& outputTest,
int const& PTrain, int const& PTest, double& rateTrain, double& rateTest)
{
    int classTrain=0, classTest=0;
    int classe;

    if(YTrain.rows()==1)
    {
            for(int p=0; p<PTrain;p++)
            {
                //std::cout << "classeTrain: " << AsTrain[L](0,p) << std::endl;
                if(outputTrain(0,p)<0.5 && round(YTrain(0,p))==0){classTrain++;}
                else if(outputTrain(0,p)>0.5 && round(YTrain(0,p))==1){classTrain++;}
            }
            for(int p=0; p<PTest;p++)
            {
                //std::cout << "classeTest: " << AsTest[L](0,p) << std::endl;
                if(outputTest(0,p)<0.5 && round(YTest(0,p))==0){classTest++;}
                else if(outputTest(0,p)>0.5 && round(YTest(0,p))==1){classTest++;}
            }
    }
    else
    {
        for(int p=0; p<PTrain;p++)
        {
            outputTrain.col(p).maxCoeff(&classe);
            if(YTrain(classe,p)==1){classTrain++;}
        }
        for(int p=0; p<PTest;p++)
        {
            outputTest.col(p).maxCoeff(&classe);
            if(YTest(classe,p)==1){classTest++;}
        }
    }

    //std::cout << "classTrain: " << classTrain << std::endl;
    rateTrain = double(classTrain)/double(PTrain);
    rateTest = double(classTest)/double(PTest);
}
