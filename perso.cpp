#include "perso.h"

std::map<std::string,double> LC_Mechanic(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_Mechanic_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), moment1 = Eigen::VectorXd::Zero(N), moment1Prec;

    double gradientNorm;
    double Em_count=0;
    double cost, E, EPrec, E0, gE, gE0, vsquarePrec=0, vsquare;
    bool condition;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double const seuilE=0.0, lambda=0.5;
    double learning_rate;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); E=cost; E0=E; if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm(); learning_rate = learning_rate_init; gE0 = gradientNorm;

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //double x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
    }

    while(gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        moment1Prec = moment1; vsquarePrec=vsquare;
        EPrec=E; gE = std::sqrt(gradientNorm*gradientNorm+vsquare);
        learning_rate = std::min(learning_rate,std::sqrt(E0/E));

        do
        {   
            moment1 = (1-learning_rate*std::sqrt(E/E0))*moment1-learning_rate*gradient;
            update(L,nbNeurons,globalIndices,weights,bias,learning_rate*moment1);
            vsquare = moment1.squaredNorm();
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte); E = cost+0.5*vsquare;
            condition=(E-EPrec>-lambda*learning_rate*std::sqrt(E/E0)*vsquare);
            //condition=(std::sqrt(E)-std::sqrt(EPrec)>-0.5*lambda*learning_rate*vsquare/std::sqrt(E0));
            if(condition)
            {
                //learning_rate = rho*learning_rate*(-(lambda-1)*beta_bar*vsquare)/(std::max((E-EPrec)/learning_rate+beta_bar*vsquare,-eps0*(lambda-1)*beta_bar*vsquare));
                learning_rate/=2;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                moment1 = moment1Prec;
            }
            iterLoop++;
        }while(condition);
        //std::cout << "iterLoop: " << iterLoop << std::endl; iterLoop=0;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        //std::cout << "lrAvant: " << learning_rate << std::endl; 
        learning_rate*=10000;
        //learning_rate=std::sqrt(E0/E);
        //learning_rate=std::min(10000*learning_rate,std::sqrt(E0/E));
        //std::cout << "lrAprès: " << learning_rate << std::endl;
        //learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(std::max((E-EPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));
        //std::cout << learning_rate << std::endl;

        std::cout << "E: " << E << std::endl;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();

        if(E-EPrec>0)
        {
            Em_count+=1;
        }

        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}

    return study;

}

std::map<std::string,double> LC_M(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_M_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), moment1 = Eigen::VectorXd::Zero(N), moment1Prec(N);

    double gradientNorm;
    double const beta_bar=2;
    double cost, E, EPrec, vsquare;
    bool condition;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double const learning_rate_max=1/beta_bar, lambda=1/(4*beta_bar);
    double learning_rate = std::min(learning_rate_init,learning_rate_max);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); E=cost; if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm();

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

    while(gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        moment1Prec = moment1;
        EPrec=E;

        learning_rate=std::min(learning_rate,learning_rate_max);
        do
        {   
            moment1 = (1-beta_bar*learning_rate)*moment1+learning_rate*gradient;
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);
            vsquare = moment1.squaredNorm();
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte); E = cost+0.5*vsquare;
            condition=(E-EPrec>-beta_bar*lambda*learning_rate*vsquare);
            if(condition)
            {
                learning_rate/=2;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                moment1 = moment1Prec;
            }
            iterLoop++;
        }while(condition);
        //std::cout << "iterLoop: " << iterLoop << std::endl; iterLoop=0;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        learning_rate*=10000;

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();
        std::cout << "gradientNorm: " << gradientNorm << std::endl;

        epoch++;
        
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> LC_RMSProp(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_rms_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_rms_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_rms_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_rms_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_rms_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), update_vector(N);
    Eigen::ArrayXd moment2 = Eigen::ArrayXd::Zero(N), moment2Prec(N);

    double gradientNorm;
    double const beta_bar=10;
    double cost, costPrec, Vdot;
    bool condition;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double const learning_rate_max=1/beta_bar, lambda=0.5;
    double const epsilon_a = std::pow(10,-1);
    double learning_rate = std::min(learning_rate_init,learning_rate_max);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm();

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

    while(gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        moment2Prec = moment2;
        costPrec=cost;

        learning_rate=std::min(learning_rate,learning_rate_max);
        do
        {   
            moment2 = (1-beta_bar*learning_rate)*moment2 + beta_bar*learning_rate*gradient.array().pow(2);
            update_vector = gradient.array()*(moment2.sqrt()+epsilon_a).inverse(); Vdot = update_vector.dot(gradient);
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*update_vector);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*Vdot);
            if(condition)
            {
                learning_rate/=2;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                moment2 = moment2Prec;
            }
            iterLoop++;
        }while(condition);
        //std::cout << "iterLoop: " << iterLoop << std::endl; iterLoop=0;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        learning_rate*=10000;

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();
        std::cout << "gradientNorm: " << gradientNorm << std::endl;

        epoch++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);

    return study;

}

std::map<std::string,double> LC_clipping(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_clipping_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec;
    bool condition;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double seuilE=0.0;
    double const gamma=10*eps, lambda=0.5;
    double learning_rate = learning_rate_init;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm();

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //double x=-4, y=3;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;

        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*std::min(1.0,gamma/gradientNorm)*gradient);
            //update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient/gradientNorm);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*std::min(1.0,gamma/gradientNorm)*std::pow(gradientNorm,2));
            //condition=(cost-costPrec>-lambda*learning_rate*gradientNorm);
            if(condition)
            {
                //learning_rate*=rho*(-(lambda-1)*gradientNorm)/(std::max((cost-costPrec)/learning_rate+gradientNorm,-eps0*(lambda-1)*gradientNorm));
                learning_rate/=2;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        //std::cout << "iterLoop: " << iterLoop << std::endl; iterLoop=0;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        std::cout << "lrAvant: " << learning_rate << std::endl; 
        //learning_rate*=rho*(-(lambda-1)*gradientNorm)/(std::max((cost-costPrec)/learning_rate+gradientNorm,-eps0*(lambda-1)*gradientNorm));
        learning_rate*=10000;
        //learning_rate = learning_rate_init;
        //std::cout << "lrAprès: " << learning_rate << std::endl;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}

    return study;

}

std::map<std::string,double> LC_signGD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_signGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec;
    bool condition;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double seuilE=0.0;
    double const lambda=0.5;
    double learning_rate = learning_rate_init;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm();

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //double x=-4, y=3;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;

        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient.array().sign());
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*gradient.lpNorm<1>());
            if(condition)
            {
                //learning_rate*=rho*(-(lambda-1)*gradientNorm)/(std::max((cost-costPrec)/learning_rate+gradientNorm,-eps0*(lambda-1)*gradientNorm));
                learning_rate/=2;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        //std::cout << "iterLoop: " << iterLoop << std::endl; iterLoop=0;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        std::cout << "lrAvant: " << learning_rate << std::endl; 
        //learning_rate*=rho*(-(lambda-1)*gradientNorm)/(std::max((cost-costPrec)/learning_rate+gradientNorm,-eps0*(lambda-1)*gradientNorm));
        learning_rate*=10000;
        //learning_rate = learning_rate_init;
        //std::cout << "lrAprès: " << learning_rate << std::endl;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}

    return study;

}

std::map<std::string,double> LC_EM(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, 
double const& learning_rate_init, double const& beta1_init, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LC_Em_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_LC_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, l;
    double beta_bar = beta1_init/learning_rate_init;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N);

    double gradientNorm=1000;
    double Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    double vsquare, prod, cost, costPrec;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double learning_rate = learning_rate_init;
    double const seuil=0.0, f1=2, f2=10000;
    double a,b,c, delta, x1,x2,x;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costPrec=cost;
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

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        vsquare=moment1.squaredNorm(); costPrec=cost; 
        if(vsquare<std::pow(10,-12))
        {
            learning_rate=learning_rate_init; 
            moment1=-beta_bar*gradient;
            update(L,nbNeurons,globalIndices,weights,bias,learning_rate*moment1);
        }
        else
        {
            prod=moment1.dot(gradient);
            do
            {
                update(L,nbNeurons,globalIndices,weights,bias,learning_rate*moment1);
                fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
                a = (2+beta_bar*learning_rate)*(4*vsquare+std::pow(beta_bar*learning_rate*gradientNorm,2)-4*beta_bar*learning_rate*prod);
                b = 2*beta_bar*learning_rate*prod-4*vsquare;
                c = beta_bar*(cost-costPrec);
                delta = b*b-4*a*c;
                if(delta<0 || std::abs(a)<std::pow(10,-12))
                {
                    learning_rate/=f1; 
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                }
            }while(delta<0 || std::abs(a)<std::pow(10,-12));
            x1=(-b-std::sqrt(delta))/(2*a); x2=(-b+std::sqrt(delta))/(2*a);
            if(std::abs(x1)<std::abs(x2)){x=x2;}
            else{x=x1;}
            moment1 = (4*x-1)*moment1-2*beta_bar*learning_rate*x*gradient;
        }
        std::cout << "eta_avant: " << learning_rate << std::endl;
        learning_rate*=f2;
        //std::cout << "learning_rate: " << learning_rate << std::endl;

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm=gradient.norm();
        std::cout << "epoch: " << epoch << std::endl;
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch); study["prop_initial_ineq"] = prop_initial_ineq/double(epoch);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/double(epoch);}

    return study;

}

std::map<std::string,double> LC_EGD2(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_EGD2_"+fileExtension+".csv").c_str());
    //std::ofstream evalFlux(("Record/eval_LC_EGD2_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_iterLoop=1, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec, V_dot;
    bool condition;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double seuilE=0.0;
    double f1=30, f2=10000, rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    double gauche, droite, m, m_best;
    int const nLoops=3; bool last_pass=false; bool active_other=false;
    double learning_rate = learning_rate_init;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    V_dot=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot);

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    double x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;

        //learning_rate=std::min(learning_rate,learning_rate_max);
        iterLoop=0;
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*V_dot);
            if(condition)
            {
                learning_rate/=f1;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);

        if(iterLoop>1)
        {
            gauche = std::log10(learning_rate); droite = std::log10(f1*learning_rate);
            for (int k=0; k<nLoops; k++)
            {
                m=(gauche+droite)/2; learning_rate=std::pow(10,m);
                update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
                fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
                if(cost-costPrec>-lambda*learning_rate*V_dot)
                {
                    m_best=gauche;
                    droite=m;
                    last_pass=false;
                }
                else
                {
                    gauche=m;
                    last_pass=true;
                }
                if(k<nLoops-1)
                {
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                }
                else
                {
                    if(!last_pass)
                    {
                        learning_rate=std::pow(10,m_best);
                        std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
                        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
                    }
                }
                iterLoop++;
            }
        }
        total_iterLoop+=iterLoop;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        if(learning_rate*gradientNorm<__DBL_EPSILON__){active_other=true; break;}
        //std::cout << learning_rate << std::endl;
        learning_rate*=f2;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        /* std::cout << "gradnorm: " << gradientNorm << std::endl;
        std::cout << "R: " << cost << std::endl; */

        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;

    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,0.9,0.999,eps,maxEpoch,tracking,record,fileExtension);
    }

    if(!active_other){
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time); study["total_iterLoop"]=double(total_iterLoop);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["epoch"]+Em_count)/(study["epoch"]+double(epoch));}
        study["epoch"]+=double(epoch); study["time"]+=double(time); study["total_iterLoop"]=double(total_iterLoop);
    }

    return study;

}

std::map<std::string,double> LC_EGD3(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_EGD3_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec, V_dot;
    bool condition;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double seuilE=0.0;
    double const rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    double learning_rate = learning_rate_init;
    bool active_other=false;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    V_dot=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot);

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //double x=-4, y=3;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;

        //learning_rate=std::min(learning_rate,learning_rate_max);
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*V_dot);
            if(condition)
            {
                learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(std::max((cost-costPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition && iterLoop<500);
        iterLoop=0;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        if(std::isnan(gradientNorm) or learning_rate<std::pow(10,-10)){active_other=true; break;}

        learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(std::max((cost-costPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            //normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        epoch++;

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,double> study;
    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,1-0.9,1-0.999,eps,maxEpoch,tracking,record,fileExtension);
    }

    if(!active_other){
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["epoch"]+Em_count)/(study["epoch"]+double(epoch));}
        study["epoch"]+=double(epoch); study["time"]+=double(time);
    }

    return study;

}

std::map<std::string,double> LC_EGD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_EGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec, V_dot;
    bool condition, active_other=false;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double seuilE=0.0, learning_rate_max=1000, rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    double learning_rate = std::min(learning_rate_init,learning_rate_max);
    double const f1=2, f2=10000;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    V_dot=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot);

    //std::cout << "cInit: " << cost << std::endl;
    std::cout << "grInit: " << gradientNorm << std::endl;

    double x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
    }

    while(gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;

        //learning_rate=std::min(learning_rate,learning_rate_max);
        iterLoop=0;
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*V_dot);
            if(condition)
            {
                //learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(std::max((cost-costPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));
                learning_rate/=f1;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        total_iterLoop+=iterLoop;
        //std::cout << "iterLoop: " << iterLoop << std::endl; iterLoop=0;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        if(learning_rate*gradientNorm<__DBL_EPSILON__){active_other=true; break;}

        //std::cout << "LAvant: " << (2*(1-lambda))/learning_rate << std::endl;
        learning_rate*=f2;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;

    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,1-0.9,1-0.999,eps,maxEpoch,tracking,record,fileExtension);
    }

    if(!active_other){
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time); study["total_iterLoop"]=double(total_iterLoop);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["epoch"]+Em_count)/(study["epoch"]+double(epoch));}
        study["epoch"]+=double(epoch); study["time"]+=double(time); study["total_iterLoop"]=double(total_iterLoop);
    }

    return study;

}

std::map<std::string,double> LCI_EGD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LCI_EGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_iterLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec, costInit, grad_square, V_dot=0;
    bool condition, active_other=false;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double seuilE=0.0, learning_rate_max=1000, rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    double learning_rate = std::min(learning_rate_init,learning_rate_max);
    double const f1=2, f2=10000;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit=cost; if(record){costsFlux << cost << std::endl;}
    grad_square = gradient.squaredNorm(); V_dot=grad_square; gradientNorm=std::sqrt(grad_square);

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    double x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
    }

    while(gradientNorm>eps && epoch<maxEpoch)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;

        //learning_rate=std::min(learning_rate,learning_rate_max);
        iterLoop=0;
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costInit>-lambda*learning_rate*V_dot);
            if(condition)
            {
                //learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(std::max((cost-costPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));
                learning_rate/=f1;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        total_iterLoop+=iterLoop;
        //std::cout << "iterLoop: " << iterLoop << std::endl;
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

        if(learning_rate*gradientNorm<__DBL_EPSILON__){active_other=true; break;}

        //std::cout << "lrAvant: " << learning_rate << std::endl;
        learning_rate*=f2;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            normeFlux << std::sqrt(std::pow(weights[0](0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        grad_square = gradient.squaredNorm(); V_dot+=grad_square; gradientNorm=std::sqrt(grad_square);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;

    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,1-0.9,1-0.999,eps,maxEpoch,tracking,record,fileExtension);
    }

    if(!active_other){
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time); study["total_iterLoop"]=double(total_iterLoop);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["epoch"]+Em_count)/(study["epoch"]+double(epoch));}
        study["epoch"]+=double(epoch); study["time"]+=double(time); study["total_iterLoop"]=double(total_iterLoop);
    }

    return study;

}

std::map<std::string,double> EulerRichardson(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_EulerRichardson_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_EulerRichardson_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_EulerRichardson_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd gradientInter = Eigen::VectorXd::Zero(N);


    double gradientNorm = 1000;
    double learning_rate = learning_rate_init;
    double erreur;

    double prop_entropie=0, prop_initial_ineq=0, modif=0, seuilE=0.01;
    double costInit,cost,costPrec;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    while (gradientNorm>eps && epoch<maxEpoch)
    {
        //std::cout << "cost: " << cost << std::endl;

       if(epoch==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(tracking){cost = risk(Y,P,As[L],type_perte); costInit = cost;}

            if(record)
            {
                if(tracking){costsFlux << cost << std::endl;}
                gradientNorm = gradient.norm(); gradientNormFlux << gradientNorm << std::endl;
            }

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        }

        erreur=(0.5*learning_rate*(gradient-gradientInter).norm())/seuil;

        if(erreur>1)
        {
           learning_rate*=0.9/std::sqrt(erreur);
        }
        else
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradientInter);

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            if(tracking)
            {
                modif++;
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
                if(!std::signbit((cost-costInit))){prop_initial_ineq++;}

                if(record){speedFlux << ((cost-costPrec)/learning_rate + gradient.squaredNorm()) << std::endl;}
            }
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            //std::cout << learning_rate << std::endl;
            learning_rate*=0.9/std::sqrt(erreur);
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        gradientNorm = gradientInter.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;

        epoch++;
        

        if(record)
        {
            if(tracking){costsFlux << cost << std::endl;}
            gradientNormFlux << gradientNorm << std::endl;
        }

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}

//Diminution naive du pas pour GD
std::map<std::string,double> GD_Em(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_GD_Em_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_GD_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, epochTot=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm=1000;
    double Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec,costInit;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double learning_rate = 0.1;
    double seuilE=0.0, facteur=1.5, factor2=1.5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
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

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;
        learning_rate=0.1;
        do
        {
            //epochTot++;
            //if(epochTot<10){std::cout << learning_rate << std::endl;}
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

            cost = risk(Y,P,As[L],type_perte);
            //std::cout << Em << std::endl;
            if(cost-costPrec>0){learning_rate/=facteur;std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());}
        }while(cost-costPrec>0);
        //std::cout << "learning_rate: " << learning_rate << std::endl;


        if(cost-costPrec>0)
        {
            Em_count+=1;
        }
        if(cost-costInit>0)
        {
            prop_initial_ineq+=1;
        }
        if(record){speedFlux << ((cost-costPrec)/learning_rate + gradient.squaredNorm()) << std::endl;}

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch); study["prop_initial_ineq"] = prop_initial_ineq/double(epoch);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/double(epoch);}

    return study;

}

std::map<std::string,double> LM_ER(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& mu_init,
double const& seuil, double const& eps, int const& maxEpoch, bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LM_ER_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;

    double prop_entropie=0, prop_initial_ineq=0, modif=0;
    double costInit, cost, costPrec;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);

     std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);

    Eigen::VectorXd gradient(N), gradientInter(N), delta(N), deltaInter(N);
    Eigen::MatrixXd Q(N,N), QInter(N,N);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N,N);

    double gradientNorm = 1000;
    double h = 1/mu_init;
    double mu = mu_init;
    double erreur;

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
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            solve(gradient,Q+mu*I,delta);
            if(tracking){cost = risk(Y,P,As[L],type_perte); costInit = cost;}

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,h*delta);
            fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,QInter,type_perte);
            solve(gradientInter,QInter+mu*I,deltaInter);
        }

        erreur=(0.5*h*(delta-deltaInter).norm())/seuil;

        if(erreur>1)
        {
           h*=0.9/std::sqrt(erreur);
        }
        else
        {
            update(L,nbNeurons,globalIndices,weights,bias,h*deltaInter);
            h*=0.9/std::sqrt(erreur);

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            solve(gradient,Q+mu*I,delta);

            if(tracking)
            {
                modif++;
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if(std::signbit((cost-costPrec))){prop_entropie++;}
                if(std::signbit((cost-costInit))){prop_initial_ineq++;}
            }
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

        update(L,nbNeurons,globalIndices,weightsInter,biasInter,h*delta);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,QInter,type_perte);
        solve(gradientInter,QInter+mu*I,deltaInter);


        gradientNorm = gradientInter.norm();

        

        epoch++;
    }

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}


//Diminution naive du pas
std::map<std::string,double> Momentum_Em(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& beta1_init, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_Em_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_Momentum_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, epochTot=0, l;
    double beta_bar = beta1_init/learning_rate_init;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd moment1 = Eigen::VectorXd::Zero(N), moment1Prec(N);

    double gradientNorm=1000;
    double Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    double EmPrec,Em,cost,costInit;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double learning_rate = 0.1, beta1 = beta1_init;
    double seuil=0.0, facteur=1.5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
    Em = beta_bar*cost;
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

        moment1Prec=moment1; std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        EmPrec=Em;
        learning_rate=0.1; beta1=beta1_init;
        do
        {
            //epochTot++;
            //if(epochTot<10){std::cout << learning_rate << std::endl;}
            moment1 = (1-beta1)*moment1 + beta1*gradient;
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);

            cost = risk(Y,P,As[L],type_perte);
            Em = 0.5*moment1.squaredNorm()+beta_bar*cost;
            //std::cout << Em << std::endl;
            if(Em-EmPrec>0){learning_rate/=facteur; beta1/=facteur; std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin()); moment1=moment1Prec;}
        }while(Em-EmPrec>0);
        //std::cout << "learning_rate: " << learning_rate << std::endl;


        if(Em-EmPrec>0)
        {
            Em_count+=1;
        }
        if(cost-costInit>0)
        {
            prop_initial_ineq+=1;
        }
        if(record){speedFlux << ((Em-EmPrec)/learning_rate+beta_bar*moment1.squaredNorm()) << std::endl;}

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm=gradient.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch); study["prop_initial_ineq"] = prop_initial_ineq/double(epoch);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/double(epoch);}

    return study;

}

std::map<std::string,double> train_Perso(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, double const& beta1, double const& beta2, int const& batch_size, double const& mu_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="EulerRichardson")
    {
        study = EulerRichardson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_EGD")
    {
        study = LC_EGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LCI_EGD")
    {
        study = LCI_EGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_EGD2")
    {
        study = LC_EGD2(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_M")
    {
        study = LC_M(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_rms")
    {
        study = LC_RMSProp(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_Mechanic")
    {
        study = LC_Mechanic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_clipping")
    {
        study = LC_clipping(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_signGD")
    {
        study = LC_signGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_EGD3")
    {
        study = LC_EGD3(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_EM")
    {
        study = LC_EM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxEpoch,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="GD_Em")
    {
        study = GD_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Momentum_Em")
    {
        study = Momentum_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxEpoch,tracking,track_continuous,record,fileExtension);
    }


    return study;
}
