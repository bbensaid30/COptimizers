#include "Sclassic.h"

std::map<std::string,Sdouble> SGD(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SGD"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l, batch;
    assert(batch_size<=P);

    Eigen::SMatrixXd echantillonX, echantillonY;
    Sdouble gradientNorm;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd gradient_tot = Eigen::SVectorXd::Zero(N);

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
        
        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);

    return study;

}

std::map<std::string,Sdouble> SGD_AllBatches(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SGD_AllBatches"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    //std::cout << "reste: " << reste_batch << std::endl;
    int number_data = batch_size, indice_begin;

    Eigen::SMatrixXd X_permut, Y_permut;
    Eigen::SMatrixXd echantillonX, echantillonY;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd gradient_tot = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm;
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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

        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);

    return study;

}

std::map<std::string,Sdouble> SAdam(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SAdam"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l, batch;
    assert(batch_size<=P);

    Eigen::SMatrixXd echantillonX, echantillonY;
    Sdouble gradientNorm;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);
    Sdouble lr_adaptive;
    Eigen::SVectorXd gradient_tot = Eigen::SVectorXd::Zero(N);

    Sdouble const epsilon_a = std::pow(10,-7);

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

        moment1 = (1-beta1)*moment1 + beta1*gradient;
        moment2 = (1-beta2)*moment2.array() + beta2*gradient.array().pow(2);
        lr_adaptive = learning_rate*Sstd::sqrt(1-Sstd::pow(1-beta2,iter+1))/(1-Sstd::pow(1-beta1,iter+1));
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
        
        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);

    return study;

}

std::map<std::string,Sdouble> SAdam_AllBatches(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_SAdam_AllBatches"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), entriesX=nbNeurons[0], entriesY=nbNeurons[L], iter=0, l, batch;
    assert(batch_size<=P);
    int const number_batch = P/batch_size,  reste_batch = P-batch_size*number_batch;
    //std::cout << "reste: " << reste_batch << std::endl;
    int number_data = batch_size, indice_begin;

    Eigen::SMatrixXd X_permut, Y_permut;
    Eigen::SMatrixXd echantillonX, echantillonY;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);
    Sdouble lr_adaptive;
    Eigen::SVectorXd gradient_tot = Eigen::SVectorXd::Zero(N);

    Sdouble const epsilon_a = std::pow(10,-7);

    Sdouble gradientNorm;
    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
    gradientNorm=gradient_tot.norm();

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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
            lr_adaptive = learning_rate*Sstd::sqrt(1-Sstd::pow(1-beta2,iter+1))/(1-Sstd::pow(1-beta1,iter+1));
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
            lr_adaptive = learning_rate*Sstd::sqrt(1-Sstd::pow(1-beta2,iter+1))/(1-Sstd::pow(1-beta1,iter+1));
            update(L,nbNeurons,globalIndices,weights,bias,-lr_adaptive*moment1.array()*(moment2.sqrt()+epsilon_a).inverse());
        }
        As[0]=X;
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient_tot,type_perte);
        gradientNorm=gradient_tot.norm();

        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);

    return study;

}

std::map<std::string,Sdouble> SAdam_WB(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, unsigned const Sseed, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_SAdam_WB"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l, batch;
    assert(batch_size<=P);

    Eigen::SMatrixXd echantillonX, echantillonY;
    Sdouble gradientNorm;

    std::mt19937 eng(Sseed);

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(P, 0, P-1);

    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);
    Eigen::SArrayXd moment2 = Eigen::SArrayXd::Zero(N);
    Eigen::SVectorXd gradient_tot = Eigen::SVectorXd::Zero(N);

    Sdouble const epsilon_a = std::pow(10,-7);

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
        
        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    Sdouble cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);

    return study;

}

std::map<std::string,Sdouble> train_Sclassic(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate, int const& batch_size, unsigned const Sseed, Sdouble const& beta1, Sdouble const& beta2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="SGD")
    {
        study = SGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,Sseed,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="SGD_AllBatches")
    {
        study = SGD_AllBatches(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,batch_size,Sseed,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="SAdam")
    {
        study = SAdam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,beta1,beta2,batch_size,Sseed,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="SAdam_AllBatches")
    {
        study = SAdam_AllBatches(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,beta1,beta2,batch_size,Sseed,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="SAdam_WB")
    {
        study = SAdam_WB(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate,beta1,beta2,batch_size,Sseed,eps,maxIter,tracking,record,fileExtension);
    }

    return study;

}

