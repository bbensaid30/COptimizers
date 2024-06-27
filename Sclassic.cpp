#include "Sclassic.h"

//ne pas oublier de faire les modifs maxIter vers maxEpoch

std::map<std::string,double> SGD(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SGD"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iter=0, l, batch;
    assert(batch_size<=P);
    int m;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}

    Eigen::MatrixXd echantillonX, echantillonY;
    double gradientNorm;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::shuffle(indices.data(), indices.data()+P, eng);
    echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
    echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
    //stochastic part
    As[0]=echantillonX;
    fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
    backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    //deterministic part
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

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

        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);

        std::shuffle(indices.data(), indices.data()+P, eng);
        echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
        //std::cout << echantillonX << std::endl;
        echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
        //stochastic part
        As[0]=echantillonX;
        fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        //deterministic part
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();
        
        iter++;
        if(iter>0 && iter%m==0){epoch++;}
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    double cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> SGD_AllBatches(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SGD_AllBatches"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], epoch=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    //std::cout << "reste: " << reste_batch << std::endl;
    int number_data = batch_size, indice_begin;

    Eigen::MatrixXd X_permut, Y_permut;
    Eigen::MatrixXd echantillonX, echantillonY;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        //pour le batch size
        std::shuffle(indices.data(), indices.data()+P, eng);
        X_permut = X*indices.asPermutation();
        Y_permut = Y*indices.asPermutation();

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

        number_data=batch_size;
        for(batch=0; batch<number_batch;batch++)
        {   
            indice_begin = batch*batch_size;

            echantillonX = X_permut.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y_permut.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X_permut.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y_permut.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
        }
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();

        epoch++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    double cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> SAdam(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SAdam"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iter=0, l, batch;
    assert(batch_size<=P);
    int m;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}

    Eigen::MatrixXd echantillonX, echantillonY;
    double gradientNorm;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N);
    double lr_adaptive;
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    double const epsilon_a = std::pow(10,-7);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::shuffle(indices.data(), indices.data()+P, eng);
    echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
    echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
    //stochastic part
    As[0]=echantillonX;
    fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
    backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    //deterministic part
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

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

        moment1 = (1-beta1)*moment1 + beta1*gradient;
        moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
        lr_adaptive = learning_rate*std::sqrt(1-std::pow(1-beta2,iter+1))/(1-std::pow(1-beta1,iter+1));
        update(L,nbNeurons,globalIndices,weights,bias,-lr_adaptive*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());

        std::shuffle(indices.data(), indices.data()+P, eng);
        echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
        echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
        //stochastic part
        As[0]=echantillonX;
        fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        //deterministic part
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();

        iter++;
        if(iter>0 && iter%m==0){epoch++;}
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    double cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> SAdam_AllBatches(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SAdam_AllBatches"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], epoch=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    //std::cout << "reste: " << reste_batch << std::endl;
    int number_data = batch_size, indice_begin;

    Eigen::MatrixXd X_permut, Y_permut;
    Eigen::MatrixXd echantillonX, echantillonY;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N);
    double lr_adaptive;
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    double const epsilon_a = std::pow(10,-7);

    double gradientNorm;
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        //pour le batch size
        std::shuffle(indices.data(), indices.data()+P, eng);
        X_permut = X*indices.asPermutation();
        Y_permut = Y*indices.asPermutation();

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

        number_data=batch_size;
        for(batch=0; batch<number_batch;batch++)
        {   
            indice_begin = batch*batch_size;

            echantillonX = X_permut.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y_permut.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
            lr_adaptive = learning_rate*std::sqrt(1-std::pow(1-beta2,epoch+1))/(1-std::pow(1-beta1,epoch+1));
            update(L,nbNeurons,globalIndices,weights,bias,-lr_adaptive*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());
        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X_permut.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y_permut.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            
            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
            lr_adaptive = learning_rate*std::sqrt(1-std::pow(1-beta2,epoch+1))/(1-std::pow(1-beta1,epoch+1));
            update(L,nbNeurons,globalIndices,weights,bias,-lr_adaptive*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());
        }
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();

        epoch++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    double cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> SAdam_WB(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SAdam_WB"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iter=0, l, batch;
    assert(batch_size<=P);
    int m;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}

    Eigen::MatrixXd echantillonX, echantillonY;
    double gradientNorm;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N);
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    double const epsilon_a = std::pow(10,-7);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::shuffle(indices.data(), indices.data()+P, eng);
    echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
    echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
    //stochastic part
    As[0]=echantillonX;
    fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
    backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    //deterministic part
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

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

        moment1 = (1-beta1)*moment1 + beta1*gradient;
        moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());

        std::shuffle(indices.data(), indices.data()+P, eng);
        echantillonX = (X*indices.asPermutation()).leftCols(batch_size);
        echantillonY = (Y*indices.asPermutation()).leftCols(batch_size);
        //stochastic part
        As[0]=echantillonX;
        fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        //deterministic part
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();

        iter++;
        if(iter>0 && iter%m==0){epoch++;}
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    double cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> SAdam_AllBatches_WB(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate, double const& beta1, double const& beta2, int const& batch_size, unsigned const Sseed, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SAdam_AllBatches_WB"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], epoch=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    //std::cout << "reste: " << reste_batch << std::endl;
    int number_data = batch_size, indice_begin;

    Eigen::MatrixXd X_permut, Y_permut;
    Eigen::MatrixXd echantillonX, echantillonY;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N);
    Eigen::VectorXd gradient_tot = Eigen::VectorXd::Zero(N);

    double const epsilon_a = std::pow(10,-7);

    double gradientNorm;
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        //pour le batch size
        std::shuffle(indices.data(), indices.data()+P, eng);
        X_permut = X*indices.asPermutation();
        Y_permut = Y*indices.asPermutation();

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

        number_data=batch_size;
        for(batch=0; batch<number_batch;batch++)
        {   
            indice_begin = batch*batch_size;

            echantillonX = X_permut.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y_permut.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());
        }
        if (reste_batch!=0)
        {
            indice_begin = number_batch*batch_size;
            number_data = reste_batch;

            echantillonX = X_permut.block(0,indice_begin,entriesX,number_data);
            echantillonY = Y_permut.block(0,indice_begin,entriesY,number_data);

            As[0]=echantillonX;
            fforward(L,number_data,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,number_data,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            
            moment1 = (1-beta1)*moment1 + beta1*gradient;
            moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());
        }
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();

        epoch++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    double cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> train_Sclassic(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate, int const& batch_size, unsigned const Sseed, double const& beta1, double const& beta2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="SGD")
    {
        study = SGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="SGDA")
    {
        study = SGD_AllBatches(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="SAdam")
    {
        study = SAdam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,beta1,beta2,batch_size,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="SAdamA")
    {
        study = SAdam_AllBatches(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,beta1,beta2,batch_size,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="SAdam_WB")
    {
        study = SAdam_WB(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,beta1,beta2,batch_size,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="SAdamA_WB")
    {
        study = SAdam_AllBatches_WB(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,beta1,beta2,batch_size,Sseed,eps,maxEpoch,tracking,record,fileExtension);
    }

    return study;

}

