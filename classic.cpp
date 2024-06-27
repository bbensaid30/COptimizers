#include "classic.h"

std::map<std::string,Sdouble> GD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_GD_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_GD_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !speedFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    assert(batch_size<=P);

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm = 1000;
    Sdouble prop_entropie=0, prop_initial_ineq=0, seuilE=0.01;
    Sdouble costInit, cost, costPrec;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }


        if(iter==0)
        {
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(tracking){cost = risk(Y,P,As[L],type_perte); costInit=cost;}
        }

        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        if(tracking)
        {
            costPrec = cost;
            cost = risk(Y,P,As[L],type_perte);
            if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
            if(!std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            if(record){speedFlux << ((cost-costPrec)/learning_rate+gradient.squaredNorm()).number << std::endl;}
        }
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm = gradient.norm();

        //std::cout << "gradientNorm: " << moyGradientNorm << std::endl;
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = prop_entropie/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> SGD_Ito(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& eps, int const& maxIter, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SGD_Ito"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    int number_data = batch_size, indice_begin;


    Eigen::SMatrixXd echantillonX, echantillonY;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble moyGradientNorm = 1000, sommeGradient;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (moyGradientNorm+std::abs(moyGradientNorm.error)>eps && iter<maxIter)
    {
        //pour le batch size
        //std::shuffle(indices.data(), indices.data()+P, eng);
        //X = X*indices.asPermutation();
        //Y = Y*indices.asPermutation();

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        sommeGradient=0;
        for(batch=0; batch<number_batch;batch++)
        {
            indice_begin = batch*batch_size;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            if (batch==0){update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);}
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();

        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            if(iter==0)
            {
                fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
                backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            }

            if (batch==0){update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);}
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            sommeGradient += gradient.norm();
        }


        moyGradientNorm = sommeGradient/number_batch;
        if(numericalNoise(moyGradientNorm)){break;}

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);

    return study;

}

std::map<std::string,Sdouble> Momentum(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_Momentum_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l, batch;
    assert(batch_size<=P);
    Sdouble beta_bar = beta1/learning_rate;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm = 1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0, seuilE=0.01;
    Sdouble EmPrec,Em,cost,costInit;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }
        }

        if(iter==0)
        {
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            if(tracking)
            {
                cost = risk(Y,P,As[L],type_perte); costInit = cost;
                Em = beta_bar*cost;
            }
        }

        moment1 = (1-beta1)*moment1 - learning_rate*gradient;
        if(track_continuous)
        {
            condition = moment1.dot(gradient);
            if(condition>=0){continuous_entropie++;}
        }

        update(L,nbNeurons,globalIndices,weights,bias,moment1);

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm = gradient.norm();

        if(tracking)
        {
            cost = risk(Y,P,As[L],type_perte);
            EmPrec = Em; Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
            if((Em-EmPrec)/EmPrec>seuilE)
            {
                   Em_count+=1;
            }
            if(cost-costInit>0)
            {
                prop_initial_ineq+=1;
            }

            if(record){speedFlux << ((Em-EmPrec)/learning_rate + beta_bar*moment1.squaredNorm()).number << std::endl;}
        }

        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> Adam_WB(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream gradientNormFlux(("Record/gradientNorm_Adam_WB_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_Adam_WB_"+fileExtension+".csv").c_str());
    if(!costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    assert(batch_size<=P);

    std::vector<Eigen::SMatrixXd> As(L+1);
    As[0]=X; // Déterministe
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);

    Sdouble gradientNorm=1000;
    Sdouble cost;

    Sdouble t=0, EmPrec,Em,Em_count=0, seuil_E=0.01;
    Sdouble beta1_bar = beta1/learning_rate, beta2_bar=beta2/learning_rate;
    Sdouble const epsilon_a = std::pow(10,-7);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        if(iter==0)
        {
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(tracking)
            {
                cost = risk(Y,P,As[L],type_perte);
                Em = beta1_bar*cost;
            }
            if(record)
            {
                if(tracking){costsFlux << cost.number << std::endl;}
                gradientNormFlux << gradient.norm().number << std::endl;
            }
        }
        t+=learning_rate;

        moment1 = (1-beta1)*moment1 + beta1*gradient;
        moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
        //update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*(moment2+std::pow(10,-10)).sqrt().inverse());

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

        if(tracking)
        {
            cost = risk(Y,P,As[L],type_perte);
            EmPrec=Em; Em = beta1_bar*cost+0.5*(moment1.array().pow(2)*((moment2+std::pow(10,-7)).rsqrt())).sum();
            if((Em-EmPrec)/EmPrec>seuil_E){Em_count++;}
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm = gradient.norm();

        //std::cout << "iter: "<< iter << " and cost: "<< cost << std::endl;
        //std::cout << "gradnorm: " << gradientNorm << std::endl;

        iter++;
        if(numericalNoise(gradientNorm)){break;}


        if(record)
        {
            if(tracking){costsFlux << cost.number << std::endl;}
            gradientNormFlux << gradientNorm.number << std::endl;
        }

    }


    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> Adam(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate,
int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream gradientNormFlux(("Record/gradientNorm_Adam_bias_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_Adam_bias_"+fileExtension+".csv").c_str());
    if(!costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;
    assert(batch_size<=P);

    std::vector<Eigen::SMatrixXd> As(L+1);
    As[0]=X; // Déterministe
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);
    Sdouble lr_adaptive;

    Sdouble gradientNorm=1000;
    Sdouble cost;

    Sdouble t=0, EmPrec,Em,Em_count=0, seuil_E=0.01;
    Sdouble beta1_bar = beta1/learning_rate, beta2_bar=beta2/learning_rate;
    Sdouble const epsilon_a = std::pow(10,-7);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        if(iter==0)
        {
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(tracking)
            {
                cost = risk(Y,P,As[L],type_perte);
                Em = beta1_bar*cost;
            }
            if(record)
            {
                if(tracking){costsFlux << cost.number << std::endl;}
                gradientNormFlux << gradient.norm().number << std::endl;
            }
        }
        t+=learning_rate;

        moment1 = (1-beta1)*moment1 + beta1*gradient;
        moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
        lr_adaptive = learning_rate*Sstd::sqrt(1-Sstd::pow(1-beta2,iter+1))/(1-Sstd::pow(1-beta1,iter+1));
        update(L,nbNeurons,globalIndices,weights,bias,-lr_adaptive*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

        if(tracking)
        {
            cost = risk(Y,P,As[L],type_perte);
            EmPrec=Em; Em = beta1_bar*cost+0.5*fAdam(beta1_bar,beta2_bar,t)*(moment1.array().pow(2)*((moment2+std::pow(10,-7)).rsqrt())).sum();
            if((Em-EmPrec)/EmPrec>seuil_E){Em_count++;}
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm = gradient.norm();

        //std::cout << "iter: "<< iter << " and cost: "<< cost << std::endl;
        //std::cout << "gradnorm: " << gradientNorm << std::endl;

        iter++;
        if(numericalNoise(gradientNorm)){break;}


        if(record)
        {
            if(tracking){costsFlux << cost.number << std::endl;}
            gradientNormFlux << gradientNorm.number << std::endl;
        }

    }


    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> train_classic(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate, Sdouble const& clip, int const& batch_size, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="GD")
    {
        study = GD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="Momentum")
    {
        study = Momentum(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Adam")
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,beta2,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="Adam_WB")
    {
        study = Adam_WB(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,beta2,eps,maxIter,tracking,record,fileExtension);
    }
    else
    {
        study = GD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxIter,record,tracking,fileExtension);
    }

    return study;

}

