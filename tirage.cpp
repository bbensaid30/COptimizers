#include "tirage.h"

std::vector<std::map<std::string,Sdouble>> tiragesRegression(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
bool const tracking)
{
    int const PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,Sdouble>> studies(nbTirages);


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
            initialisation(nbNeurons,weights,bias,supParameters,generator,i);
            studies[i] = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
            learning_rate,clip,seuil,beta1,beta2,batch_size,
            mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking);

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);
            studies[i]["cost_test"] = costTest;
            studies[i]["num_tirage"] = i;

            /* std::cout << "On est au tirage: " << i << std::endl;
            std::cout << "iters: " << studies[i]["iter"] << std::endl;
            std::cout << "costTrain: " << studies[i]["finalCost"] << std::endl;
            std::cout << "costTest: " << studies[i]["cost_test"] << std::endl;
            std::cout << "Numéro Thread: " << omp_get_thread_num() << std::endl; */
        }
    }

    return studies;

}

std::vector<std::map<std::string,Sdouble>> tiragesClassification(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
bool const tracking)
{
    int const PTrain=data[0].cols(), PTest = data[2].cols();

    int const tirageMax=tirageMin+nbTirages;

    std::vector<std::map<std::string,Sdouble>> studies(nbTirages);


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
            initialisation(nbNeurons,weights,bias,supParameters,generator,i);
            studies[i] = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
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

void minsRecordRegression(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTirages = studies.size();
    int nonConv=0, div=0, numNoise=0;

    std::map<std::string,Sdouble> study;

    for(int i=0; i<nbTirages; i++)
    {
        study = studies[i];

        if((study["finalGradient"]+std::abs(study["finalGradient"].error)<eps) && !Sstd::isnan(study["finalGradient"]) && !Sstd::isinf(study["finalGradient"]) && !numericalNoise(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"].number << std::endl;
            infosFlux << study["iter"].number << std::endl;
            infosFlux << study["time"].number << std::endl;

            infosFlux << study["finalCost"].number << std::endl;
            infosFlux << std::abs(study["finalCost"].error) << std::endl;

            infosFlux << study["cost_test"].number << std::endl;
            infosFlux << std::abs(study["cost_test"].error) << std::endl;

            infosFlux << study["prop_entropie"].number << std::endl;

            infosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;

            //------------------------------------------------------

            allinfosFlux << study["num_tirage"].number << std::endl;
            allinfosFlux << study["iter"].number << std::endl;
            allinfosFlux << study["time"].number << std::endl;

            allinfosFlux << study["finalCost"].number << std::endl;
            allinfosFlux << std::abs(study["finalCost"].error) << std::endl;

            allinfosFlux << study["cost_test"].number << std::endl;
            allinfosFlux << std::abs(study["cost_test"].error) << std::endl;

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
                std::cout << study["finalGradient"].number << std::endl;
                nonConv++;
                nonFlux << -2 << std::endl;
                nonFlux << study["prop_entropie"].number << std::endl;

                allinfosFlux << study["num_tirage"].number << std::endl;
                allinfosFlux << study["iter"].number << std::endl;
                allinfosFlux << study["time"].number << std::endl;

                allinfosFlux << study["finalCost"].number << std::endl;
                allinfosFlux << std::abs(study["finalCost"].error) << std::endl;

                allinfosFlux << study["cost_test"].number << std::endl;
                allinfosFlux << std::abs(study["cost_test"].error) << std::endl;

                allinfosFlux << study["prop_entropie"].number << std::endl;

                allinfosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;
            }
        }

    }

    infosFlux << (Sdouble(nonConv)/Sdouble(nbTirages)).number << std::endl;
    infosFlux << (Sdouble(div)/Sdouble(nbTirages)).number << std::endl;
    infosFlux << (Sdouble(numNoise)/Sdouble(nbTirages)).number << std::endl;

    allinfosFlux << (Sdouble(nonConv)/Sdouble(nbTirages)).number << std::endl;
    allinfosFlux << (Sdouble(div)/Sdouble(nbTirages)).number << std::endl;
    allinfosFlux << (Sdouble(numNoise)/Sdouble(nbTirages)).number << std::endl;

    std::cout << "Proportion de divergence: " << Sdouble(div)/Sdouble(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << Sdouble(nonConv)/Sdouble(nbTirages) << std::endl;
    std::cout << "Proportion de numerical noise: " << Sdouble(numNoise)/Sdouble(nbTirages) << std::endl;

}

void minsRecordClassification(std::vector<std::map<std::string,Sdouble>> studies, std::string const& folder, std::string const& fileEnd, Sdouble const& eps)
{

    std::ofstream infosFlux(("Record/"+folder+"/info_"+fileEnd).c_str());
    std::ofstream allinfosFlux(("Record/"+folder+"/allinfo_"+fileEnd).c_str());
    std::ofstream nonFlux(("Record/"+folder+"/nonConv_"+fileEnd).c_str());

    if(!infosFlux || !nonFlux)
    {
        std::cout << "Impossible d'ouvrir un des fichiers en écriture" << std::endl; exit(1);
    }

    int const nbTirages = studies.size();
    int nonConv=0, div=0, numNoise=0;

    std::map<std::string,Sdouble> study;

    for(int i=0; i<nbTirages; i++)
    {
        study = studies[i];

        if((study["finalGradient"]+std::abs(study["finalGradient"].error)<eps) && !Sstd::isnan(study["finalGradient"]) && !Sstd::isinf(study["finalGradient"]) && !numericalNoise(study["finalGradient"]))
        {

            infosFlux << study["num_tirage"].number << std::endl;
            infosFlux << study["iter"].number << std::endl;
            infosFlux << study["time"].number << std::endl;

            infosFlux << study["finalCost"].number << std::endl;
            infosFlux << std::abs(study["finalCost"].error) << std::endl;

            infosFlux << study["cost_test"].number << std::endl;
            infosFlux << std::abs(study["cost_test"].error) << std::endl;

            infosFlux << study["prop_entropie"].number << std::endl;

            infosFlux << study["classTrain"].number << std::endl;
            infosFlux << study["classTest"].number << std::endl;

            infosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;

            //---------------------------------------------------

            allinfosFlux << study["num_tirage"].number << std::endl;
            allinfosFlux << study["iter"].number << std::endl;
            allinfosFlux << study["time"].number << std::endl;

            allinfosFlux << study["finalCost"].number << std::endl;
            allinfosFlux << std::abs(study["finalCost"].error) << std::endl;

            allinfosFlux << study["cost_test"].number << std::endl;
            allinfosFlux << std::abs(study["cost_test"].error) << std::endl;

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
                allinfosFlux << std::abs(study["finalCost"].error) << std::endl;

                allinfosFlux << study["cost_test"].number << std::endl;
                allinfosFlux << std::abs(study["cost_test"].error) << std::endl;

                allinfosFlux << study["prop_entropie"].number << std::endl;

                allinfosFlux << study["classTrain"].number << std::endl;
                allinfosFlux << study["classTest"].number << std::endl;

                allinfosFlux << study["total_iterLoop"].number/study["iter"].number << std::endl;
            }
        }

    }

    infosFlux << (Sdouble(nonConv)/Sdouble(nbTirages)).number << std::endl;
    infosFlux << (Sdouble(div)/Sdouble(nbTirages)).number << std::endl;
    infosFlux << (Sdouble(numNoise)/Sdouble(nbTirages)).number << std::endl;

    allinfosFlux << (Sdouble(nonConv)/Sdouble(nbTirages)).number << std::endl;
    allinfosFlux << (Sdouble(div)/Sdouble(nbTirages)).number << std::endl;
    allinfosFlux << (Sdouble(numNoise)/Sdouble(nbTirages)).number << std::endl;

    std::cout << "Proportion de divergence: " << Sdouble(div)/Sdouble(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << Sdouble(nonConv)/Sdouble(nbTirages) << std::endl;
    std::cout << "Proportion de numerical noise: " << Sdouble(numNoise)/Sdouble(nbTirages) << std::endl;

}

void predictionsRecord(std::vector<Eigen::SMatrixXd>& data, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& famille_algo, std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eps, int const& maxIter,
Sdouble const& learning_rate, Sdouble const& clip, Sdouble const& seuil, Sdouble const& beta1, Sdouble const& beta2, int const & batch_size,
Sdouble& mu, Sdouble& factor, Sdouble const& RMin, Sdouble const& RMax, int const& b, Sdouble const& alpha,
Sdouble const& pas, Sdouble const& Rlim, Sdouble& factorMin, Sdouble const& power, Sdouble const& alphaChap, Sdouble const& epsDiag,
std::string const& folder, std::string const fileExtension, bool const tracking, bool const track_continuous)
{

    int const PTrain = data[0].cols();
    int const PTest = data[2].cols();

    std::string const fileEnd = informationFile(PTrain,PTest,L,nbNeurons,activations,type_perte,algo,supParameters,generator,tirageMin,nbTirages,learning_rate,eps,maxIter);

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
    std::vector<Eigen::SMatrixXd> weights(L);
    std::vector<Eigen::SVectorXd> bias(L);
    int const N = globalIndices[2*L-1];

    std::vector<Eigen::SMatrixXd> AsTrain(L+1); AsTrain[0]=data[0];
    std::vector<Eigen::SMatrixXd> AsTest(L+1); AsTest[0]=data[2];
    std::vector<Eigen::SMatrixXd> slopes(L);

    Eigen::SMatrixXd bestPredictionsTrain(nbNeurons[L],PTrain), bestPredictionsTest(nbNeurons[L],PTest);
    Eigen::SMatrixXd moyPredictionsTrain(nbNeurons[L],PTrain), moyPredictionsTest(nbNeurons[L],PTest);
    moyPredictionsTrain.setZero(); moyPredictionsTest.setZero();

    Sdouble costMin=10000, costTest;

    std::map<std::string,Sdouble> study;
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
        study = train(data[0],data[1],L,nbNeurons,globalIndices,activations,weights,bias,type_perte,famille_algo,algo,eps,maxIter,
        learning_rate,clip,seuil,beta1,beta2,batch_size,
        mu,factor,RMin,RMax,b,alpha,pas,Rlim,factorMin,power,alphaChap,epsDiag,tracking,track_continuous);
        if(study["finalGradient"]+std::abs(study["finalGradient"].error)<eps && !Sstd::isnan(study["finalCost"]) && !Sstd::isinf(study["finalCost"]) && !numericalNoise(study["finalGradient"]))
        {
            fforward(L,PTrain,nbNeurons,activations,weights,bias,AsTrain,slopes);

            costFlux << i << std::endl;
            costFlux << study["iter"].number << std::endl;
            costFlux << study["finalCost"].number << std::endl;
            costFlux << std::abs(study["finalCost"].error) << std::endl;

            fforward(L,PTest,nbNeurons,activations,weights,bias,AsTest,slopes);
            costTest = risk(data[3],PTest,AsTest[L],type_perte);

            costTestFlux << i << std::endl;
            costTestFlux << study["iter"].number << std::endl;
            costTestFlux << costTest.number << std::endl;
            costTestFlux << std::abs(costTest.error) << std::endl;

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
            if(Sstd::abs(study["finalGradient"])>1 || Sstd::isnan(study["finalGradient"]))
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
            trackingFlux << study["iter"].number << std::endl;
            trackingFlux << study["prop_entropie"].number << std::endl;
        }

        if(track_continuous)
        {
            trackContinuousFlux << nMin << std::endl;
            trackContinuousFlux << study["iter"].number << std::endl;
            trackContinuousFlux << study["continuous_entropie"].number << std::endl;
        }
    }

    std::cout << "Proportion de divergence: " << Sdouble(div)/Sdouble(nbTirages) << std::endl;
    std::cout << "Proportion de non convergence: " << Sdouble(nonConv)/Sdouble(nbTirages) << std::endl;

    bestTrainFlux << bestPredictionsTrain << std::endl;
    bestTestFlux << bestPredictionsTest << std::endl;
    moyTrainFlux << moyPredictionsTrain/Sdouble(minAttain) << std::endl;
    moyTestFlux << moyPredictionsTest/Sdouble(minAttain) << std::endl;

}


std::string informationFile(int const& PTrain, int const& PTest, int const& L, std::vector<int> const& nbNeurons,
std::vector<std::string> const& activations, std::string const& type_perte,
std::string const& algo, std::vector<double> const& supParameters, std::string const& generator,
int const& tirageMin, int const& nbTirages, Sdouble const& eta, Sdouble const& eps, int const& maxIter, std::string const fileExtension)
{
    std::ostringstream epsStream;
    epsStream << eps.number;
    std::string epsString = epsStream.str();
    std::ostringstream etaStream;
    etaStream << eta.number;
    std::string etaString = etaStream.str();
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


    return algo+"("+fileExtension+")"+archi+"(eta="+etaString+", eps="+epsString+", PTrain="+PTrainString+", PTest="+PTestString+", tirageMin="+tirageMinString+", nbTirages="+nbTiragesString+", maxIter="+maxIterString+")"+ gen +".csv";
}

void classificationRate(Eigen::SMatrixXd const& YTrain, Eigen::SMatrixXd const& YTest, Eigen::SMatrixXd const& outputTrain, Eigen::SMatrixXd const& outputTest,
int const& PTrain, int const& PTest, Sdouble& rateTrain, Sdouble& rateTest)
{
    int classTrain=0, classTest=0;
    int classe;

    if(YTrain.rows()==1)
    {
            for(int p=0; p<PTrain;p++)
            {
                //std::cout << "classeTrain: " << AsTrain[L](0,p) << std::endl;
                if(outputTrain(0,p)<0.5 && round(YTrain(0,p).number)==0){classTrain++;}
                else if(outputTrain(0,p)>0.5 && round(YTrain(0,p).number)==1){classTrain++;}
            }
            for(int p=0; p<PTest;p++)
            {
                //std::cout << "classeTest: " << AsTest[L](0,p) << std::endl;
                if(outputTest(0,p)<0.5 && round(YTest(0,p).number)==0){classTest++;}
                else if(outputTest(0,p)>0.5 && round(YTest(0,p).number)==1){classTest++;}
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
    rateTrain = Sdouble(classTrain)/Sdouble(PTrain);
    rateTest = Sdouble(classTest)/Sdouble(PTest);
}
