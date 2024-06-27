#include "classic.h"

std::map<std::string,double> GD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_GD_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_GD_"+fileExtension+".csv").c_str());
    if(!weightsFlux || !speedFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    assert(batch_size<=P);

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm = 1000;
    double prop_entropie=0, prop_initial_ineq=0, seuilE=0.01;
    double costInit, cost, costPrec;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
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


        if(epoch==0)
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
            if(!std::signbit((cost-costInit))){prop_initial_ineq++;}
            if(record){speedFlux << ((cost-costPrec)/learning_rate+gradient.squaredNorm()) << std::endl;}
        }
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm = gradient.norm();

        epoch++;

    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = prop_entropie/double(epoch); study["prop_initial_ineq"] = prop_initial_ineq/double(epoch);}

    return study;

}

std::map<std::string,double> Momentum(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta1, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_Momentum_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l, batch;
    assert(batch_size<=P);
    double beta_bar = beta1/learning_rate;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);

    double gradientNorm = 1000;
    double Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0, seuilE=0.01;
    double EmPrec,Em,cost,costInit;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
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

        if(epoch==0)
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

            if(record){speedFlux << ((Em-EmPrec)/learning_rate + beta_bar*moment1.squaredNorm()) << std::endl;}
        }

        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        epoch++;

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch); study["prop_initial_ineq"] = prop_initial_ineq/double(epoch);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/double(epoch);}

    return study;

}

std::map<std::string,double> RMSProp(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream gradientNormFlux(("Record/gradientNorm_rms_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_rms_"+fileExtension+".csv").c_str());
    if(!costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    assert(batch_size<=P);

    std::vector<Eigen::MatrixXd> As(L+1);
    As[0]=X; // Déterministe
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N);

    double gradientNorm=1000;
    double cost;

    double beta2_bar=beta2/learning_rate;
    double const epsilon_a = std::pow(10,-7);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        if(epoch==0)
        {
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        }

        moment2 = (1-beta2)*moment2 + beta2*gradient.array().pow(2);
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient.array()*(moment2.sqrt()+epsilon_a).inverse());

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm = gradient.norm();

        epoch++;

        if(record)
        {
            if(tracking){costsFlux << cost << std::endl;}
            gradientNormFlux << gradientNorm << std::endl;
        }

    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> Adam_WB(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream gradientNormFlux(("Record/gradientNorm_Adam_WB_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_Adam_WB_"+fileExtension+".csv").c_str());
    if(!costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    assert(batch_size<=P);

    std::vector<Eigen::MatrixXd> As(L+1);
    As[0]=X; // Déterministe
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N);

    double gradientNorm=1000;
    double cost;

    double t=0, EmPrec,Em,Em_count=0, seuil_E=0.01;
    double beta1_bar = beta1/learning_rate, beta2_bar=beta2/learning_rate;
    double const epsilon_a = std::pow(10,-7);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        if(epoch==0)
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
                if(tracking){costsFlux << cost << std::endl;}
                gradientNormFlux << gradient.norm() << std::endl;
            }
        }
        t+=learning_rate;

        moment1 = (1-beta1)*moment1 + beta1*gradient;
        moment2 = (1-beta2)*moment2 + beta2*gradient.array().pow(2);
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

        if(tracking)
        {
            cost = risk(Y,P,As[L],type_perte);
            EmPrec=Em; Em = beta1_bar*cost+0.5*(moment1.array().pow(2)*((moment2+std::pow(10,-7)).rsqrt())).sum();
            if((Em-EmPrec)/EmPrec>seuil_E){Em_count++;}
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm = gradient.norm();

        //std::cout << "epoch: "<< epoch << " and cost: "<< cost << std::endl;
        //std::cout << "gradnorm: " << gradientNorm << std::endl;

        epoch++;


        if(record)
        {
            if(tracking){costsFlux << cost << std::endl;}
            gradientNormFlux << gradientNorm << std::endl;
        }

    }


    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}

    return study;

}

std::map<std::string,double> Adam(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
int const& batch_size, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream gradientNormFlux(("Record/gradientNorm_Adam_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_Adam_"+fileExtension+".csv").c_str());
    if(!costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;
    assert(batch_size<=P);

    std::vector<Eigen::MatrixXd> As(L+1);
    As[0]=X; // Déterministe
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N);
    double lr_adaptive;

    double gradientNorm=1000;
    double cost;

    double t=0, EmPrec,Em,Em_count=0, seuil_E=0.01;
    double beta1_bar = beta1/learning_rate, beta2_bar=beta2/learning_rate;
    double const epsilon_a = std::pow(10,-7);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        if(epoch==0)
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
                if(tracking){costsFlux << cost << std::endl;}
                gradientNormFlux << gradient.norm() << std::endl;
            }
        }
        t+=learning_rate;

        moment1 = (1-beta1)*moment1 + beta1*gradient;
        moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
        lr_adaptive = learning_rate*std::sqrt(1-std::pow(1-beta2,epoch+1))/(1-std::pow(1-beta1,epoch+1));
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

        //std::cout << "epoch: "<< epoch << " and cost: "<< cost << std::endl;
        //std::cout << "gradnorm: " << gradientNorm << std::endl;

        epoch++;

        if(record)
        {
            if(tracking){costsFlux << cost << std::endl;}
            gradientNormFlux << gradientNorm << std::endl;
        }

    }


    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}

    return study;

}

std::map<std::string,double> train_classic(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate, double const& clip, int const& batch_size, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="GD")
    {
        study = GD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="Momentum")
    {
        study = Momentum(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,eps,maxEpoch,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="rms")
    {
        study = RMSProp(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta2,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="Adam")
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,beta2,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="Adam_WB")
    {
        study = Adam_WB(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,beta1,beta2,eps,maxEpoch,tracking,record,fileExtension);
    }
    else
    {
        study = GD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,eps,maxEpoch,record,tracking,fileExtension);
    }

    return study;

}

