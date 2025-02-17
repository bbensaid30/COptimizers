#include "essai.h"

std::map<std::string,double> HolderHB(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_HHB_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_HHB_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_HHB_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_HHB_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_HHB_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, k=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient=Eigen::VectorXd::Zero(N), gradientPrec=Eigen::VectorXd::Zero(N), gradientMoy=Eigen::VectorXd::Zero(N), moment1 = Eigen::VectorXd::Zero(N);

    double gradientNorm, gradientMoyNorm, normv2, sumv=0, prodscalar, h=0;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec, costBest, costMoy;

    std::vector<Eigen::MatrixXd> weightsMoy(L), weightsBest(L);
    std::vector<Eigen::VectorXd> biasMoy(L), biasBest(L);
    double const alpha=2, beta=1;
    double lip=1000;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    std::copy(weights.begin(),weights.end(),weightsBest.begin()); std::copy(bias.begin(),bias.end(),biasBest.begin());
    std::copy(weights.begin(),weights.end(),weightsMoy.begin()); std::copy(bias.begin(),bias.end(),biasMoy.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    gradientNorm=gradient.norm(); cost = risk(Y,P,As[L],type_perte); costBest=cost;
    if(record){costsFlux << cost << std::endl;}

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
        k++; epoch++;
        convexCombination(weightsMoy,biasMoy,weights,bias,L,double(k-1)/double(k));

        moment1 = moment1 - 1/lip*gradient;
        update(L,nbNeurons,globalIndices,weights,bias,moment1);
        costPrec=cost; gradientPrec=gradient;

        fforward(L,P,nbNeurons,activations,weightsMoy,biasMoy,As,slopes); costMoy=risk(Y,P,As[L],type_perte); backward(Y,L,P,nbNeurons,activations,globalIndices,weightsMoy,biasMoy,As,slopes,gradientMoy,type_perte);
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost=risk(Y,P,As[L],type_perte); backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientMoyNorm = gradientMoy.norm();
        std::cout << "gr: " << gradient.norm() << std::endl;

        if(cost<costBest)
        {
            costBest=cost;
            std::copy(weights.begin(),weights.end(),weightsBest.begin()); std::copy(bias.begin(),bias.end(),biasBest.begin());
            gradientNorm = gradient.norm();
        }
        else if(costMoy<costBest)
        {
            costBest=costMoy;
            std::copy(weightsMoy.begin(),weightsMoy.end(),weightsBest.begin()); std::copy(biasMoy.begin(),biasMoy.end(),biasBest.begin());
            gradientNorm=gradientMoyNorm;
        }

        normv2 = moment1.squaredNorm();
        sumv+=normv2;
        prodscalar = gradientPrec.dot(moment1);
        if(cost-costPrec>prodscalar+lip/2*normv2)
        {
            std::copy(weightsBest.begin(),weightsBest.end(),weights.begin()); std::copy(biasBest.begin(),biasBest.end(),bias.begin());
            moment1 = Eigen::VectorXd::Zero(N);
            lip*=alpha; k=0; sumv=0; h=0;
        }

        if(k>=1){h = std::max(h, 3/normv2*(cost-costPrec-0.5*prodscalar-0.5*gradient.dot(moment1))); h = std::max(h, std::sqrt(8/(k*sumv))*(gradientMoyNorm-lip/k*std::sqrt(normv2)));}

        if(k*(k+1)*h>3/8*lip)
        {
            std::copy(weightsBest.begin(),weightsBest.end(),weights.begin()); std::copy(biasBest.begin(),biasBest.end(),bias.begin());
            moment1 = Eigen::VectorXd::Zero(N);
            lip*=beta; k=0; sumv=0; h=0;
        }

        //learning_rate=std::min(learning_rate,learning_rate_max);

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

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,double> study;

    std::copy(weightsBest.begin(),weightsBest.end(),weights.begin()); std::copy(biasBest.begin(),biasBest.end(),bias.begin());

    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}

    return study;

}


std::map<std::string,double> LC_RK(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_RK_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_RK_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_RK_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_RK_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_RK_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_epochLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost,costPrec, V_dot1, V_dot2;
    bool condition, active_other=false;

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    double seuilE=0.0, learning_rate_max=1, rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    double learning_rate = std::min(learning_rate_init,learning_rate_max);
    double const f1=2, f2=2;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    V_dot1=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot1);

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
            update(L,nbNeurons,globalIndices,weights,bias,-0.25*learning_rate*gradient);
            fforward(L,P,nbNeurons,activations, weights, bias, As, slopes); backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte); V_dot2=gradient.squaredNorm();
            RKCombination(weights,bias,weightsPrec,biasPrec,L); update(L,nbNeurons,globalIndices,weights,bias,-0.5*learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*V_dot1);
            if(condition)
            {
                //learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(std::max((cost-costPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));
                learning_rate/=f1;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        total_epochLoop+=iterLoop;
        //std::cout << "iterLoop: " << iterLoop << std::endl; iterLoop=0;
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
        V_dot1=gradient.squaredNorm(); gradientNorm=std::sqrt(V_dot1);

        std::cout << "gNorm: " << gradientNorm << std::endl;

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
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=double(time); study["total_epochLoop"]=double(total_epochLoop);
    if(tracking){study["prop_entropie"] = Em_count/double(epoch);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["epoch"]+Em_count)/(study["epoch"]+double(epoch));}
        study["epoch"]+=double(epoch); study["time"]+=double(time); study["total_epochLoop"]=double(total_epochLoop);
    }

    return study;

}

std::map<std::string,double> RK(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_RK_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_RK_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_RK_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_RK_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_RK_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_epochLoop=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    double gradientNorm;
    double Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    double cost, costPrec;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    gradientNorm = gradient.norm();

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

        update(L,nbNeurons,globalIndices,weights,bias,-0.25*learning_rate*gradient);
        fforward(L,P,nbNeurons,activations, weights, bias, As, slopes); backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        RKCombination(weights,bias,weightsPrec,biasPrec,L); update(L,nbNeurons,globalIndices,weights,bias,-0.5*learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); costPrec=cost; cost = risk(Y,P,As[L],type_perte);
        if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;}

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
        gradientNorm = gradient.norm();

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        epoch++;
        

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm: " << gradientNorm << std::endl;
    std::map<std::string,double> study;
 
    if(tracking){study["prop_entropie"] = Em_count;}
    study["epoch"]+=double(epoch); study["time"]+=double(time); study["total_epochLoop"]=double(total_epochLoop);

    return study;

}

std::map<std::string,double> PGD(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_PGD_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_PGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PGD_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_PGD_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_PGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, epochForward=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    std::vector<Eigen::MatrixXd> weightsRetour(L);
    std::vector<Eigen::VectorXd> biasRetour(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), gradientRetour(N);


    double V_dot, inv, gradientNorm;
    double learning_rate = learning_rate_init;

    double prop_entropie=0, seuilE=0.01;
    double cost,costPrec,costInter;
    double lambda=0;
    bool projection, convergence=true;
    double eps_R = std::pow(eps,2)/10;
    double const factor1=2, factor2=1.5;
    int iterLoop=0, fastLoop=5;

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

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);
    double const x=-2, y=1; 
    if(record){costsFlux << cost << std::endl; normeFlux << std::sqrt(std::pow(weights[0](0,0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;}
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot);
    //std::cout << "gprec: " << gradientNorm.digits() << std::endl;
    //std::cout << "gr: " << gradientNorm << std::endl;
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        std::copy(weights.begin(),weights.end(),weightsRetour.begin()); std::copy(bias.begin(),bias.end(),biasRetour.begin());

        costInter = cost-learning_rate*V_dot;
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
        //if(epoch==17 || epoch==16){std::cout << "dg: " << gradient(0).digits() << std::endl;}
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        costPrec=cost; cost = risk(Y,P,As[L],type_perte);
        gradientRetour=gradient; backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        inv = gradient.squaredNorm();
        lambda=0;
        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        projection=(std::abs(cost-costInter)>eps_R);
        iterLoop=0;
        while(projection && iterLoop<1000)
        {
            lambda -= (cost-costInter)/inv;
            update(L,nbNeurons,globalIndices,weights,bias,lambda*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            cost = risk(Y,P,As[L],type_perte);
            projection=(std::abs(cost-costInter)>eps_R);
            if((iterLoop>20 && std::abs(cost-costInter)>std::pow(10,6)) || std::isnan(cost-costInter) || std::isinf(cost-costInter)){convergence=false; break;}
            //if((projection && cost>std::pow(10,6)) || std::isnan(cost) || std::isinf(cost) || numericalNoise(cost)){convergence=false; break;}
            if(projection){std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());}
            iterLoop++;
        }
        

        if(convergence && !projection)
        {
           /*  std::cout << "iterLoop: " << iterLoop << std::endl;
            std::cout << "dE: " << learning_rate*gradientNorm*gradientNorm << std::endl;
            std::cout << "gNorm: " << gradientNorm << std::endl; */
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot);
            if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;
            normeFlux << std::sqrt(std::pow(weights[0](0,0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;}
            if(iterLoop<fastLoop){learning_rate*=factor2;}
            //if(iterLoop<fastLoop && gradientNorm>std::pow(10,-2)){learning_rate*=factor2; learning_rate=std::min(learning_rate,0.5);}
            epochForward++;

            //std::cout << "cost: " << cost << std::endl;

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
        }

        else
        {
            std::copy(weightsRetour.begin(),weightsRetour.end(),weights.begin()); std::copy(biasRetour.begin(),biasRetour.end(),bias.begin());
            cost=costPrec; gradient=gradientRetour;
            learning_rate/=factor1;
            convergence=true;
        }
        //std::cout << "lr: " << learning_rate << std::endl;

        /* std::cout << "epoch: " << epoch << std::endl;
        sleep(1);
        std::cout << "gprec: " << gradientNorm.digits() << std::endl; */
        

        if(tracking)
        {
            if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
        }

        epoch++;
        

    }
    /* std::cout << "epochForward: " << epochForward << std::endl;
    std::cout << "epoch: " << epoch << std::endl;  */
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //Shaman::displayUnstableBranches();
    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["epochForward"]=double(epochForward); study["finalGradient"]=gradientNorm; 
    study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"]=prop_entropie/epoch;}

    return study;

}

std::map<std::string,double> PGD_Brent(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_PGD_Brent_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PGD_Brent_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_PGD_Brent_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_PGD_Brent_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, epochForward=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    std::vector<Eigen::MatrixXd> weightsRetour(L);
    std::vector<Eigen::VectorXd> biasRetour(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), gradientRetour(N), gradientInter(N);


    double V_dot, gradientNorm;
    double learning_rate = learning_rate_init;

    double prop_entropie=0, seuilE=0.01;
    double cost,costPrec,costInter;
    double x,dx,a,b,c,d,s,evalx,evala,evalb,evalc,evals;
    double const twosqrt=std::sqrt(2);
    bool mflag,projection, convergence=true;
    double eps_R = std::pow(eps,2)/10;
    double const factor1=2, factor2=1.5;
    int iterLoop=0, fastLoop=100;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);
    /* double const x=-2, y=1; 
    if(record){costsFlux << cost << std::endl; normeFlux << std::sqrt(std::pow(weights[0](0,0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;} */
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot);
    //std::cout << "gprec: " << gradientNorm.digits() << std::endl;
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        std::copy(weights.begin(),weights.end(),weightsRetour.begin()); std::copy(bias.begin(),bias.end(),biasRetour.begin());

        costInter = cost-learning_rate*V_dot; 
        std::cout << "costInter: " << costInter << std::endl;
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        gradientRetour=gradient; backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());

        //------------------ Construction d'un intervalle de recherche ----------------------------------------------------------------------
        x=0;
        update(L,nbNeurons,globalIndices,weights,bias,x*gradient); fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        evalx = risk(Y,P,As[L],type_perte)-costInter; 
        std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
        if(std::abs(evalx)<eps_R){b=x; evalb=evalx;}
        else if(std::isnan(evalx) || std::isinf(evalx)){convergence=false;}

        if(std::abs(x)>std::pow(10,-10)){dx=x/50;}else{dx=1/50;}
        a=x; b=x; evala=evalx; evalb=evalx;
        while(((evala>0) == (evalb>0)) && convergence && iterLoop<1000)
        {
            dx*=twosqrt;

            a = x - dx;
            update(L, nbNeurons, globalIndices, weights, bias, a * gradient);
            fforward(L, P, nbNeurons, activations, weights, bias, As, slopes);
            evala = risk(Y, P, As[L], type_perte) - costInter;
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            if (std::isnan(evala) || std::isinf(evala)){convergence = false; break;}

            if((evala>0) != (evalb>0)){break;}

            b = x + dx;
            update(L, nbNeurons, globalIndices, weights, bias, b * gradient);
            fforward(L, P, nbNeurons, activations, weights, bias, As, slopes);
            evalb = risk(Y, P, As[L], type_perte) - costInter;
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            if (std::isnan(evalb) || std::isinf(evalb)){convergence = false; break;}

            iterLoop++;

        }

        //------------------ Méthode de Brent ----------------------------------------------------------------------------

        if(std::abs(evala)<std::abs(evalb)){echanger(a,b); echanger(evala,evalb);}
        c=a; evalc=evala; mflag=true;

        projection=(std::abs(evalb)>eps_R && std::abs(b-a)>eps_R);
        while(convergence && projection && iterLoop<1000)
        {
            if(std::abs(evala-evalc)>std::pow(10,-10) && std::abs(evalb-evalc)>std::pow(10,-10))
            {
                s = (a*evalb*evalc)/((evala-evalb)*(evala-evalc)) + (b*evala*evalc)/((evalb-evala)*(evalb-evalc)) + (c*evala*evalb)/((evalc-evala)*(evalc-evalb));            
            }
            else
            {
                s = b-(evalb*(b-a))/(evalb-evala);
            }

            if(appartient_intervalle(s,(3*a+b)/4,b) || (mflag==true && std::abs(s-b)>=std::abs(b-c)/2) || (mflag==false && std::abs(s-b)>=std::abs(c-d)/2))
            {
                s=(a+b)/2; mflag=true;
            }
            else{mflag=false;}

            update(L,nbNeurons,globalIndices,weights,bias,s*gradient); fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            evals = risk(Y,P,As[L],type_perte)-costInter; 
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            d=c; c=b; evalc=evalb;
            if(evala*evals<0){b=s; evalb=evals;}
            else{a=s; evala=evals;}
            if(std::abs(evala)<std::abs(evalb)){echanger(a,b); echanger(evala,evalb);}

            iterLoop++;
            projection=(std::abs(evalb)>eps_R && std::abs(b-a)>eps_R);
        }

        //------------------ Fin de la méthode de Brent --------------------------------------------------------------------------



        if(convergence && !projection)
        {
            update(L,nbNeurons,globalIndices,weights,bias,b*gradient);
            cost=costInter;

            //std::cout << "iterLoop: " << iterLoop << std::endl;
            //std::cout << "dE: " << learning_rate*gradientNorm*gradientNorm << std::endl;
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot);
            /* if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;
            normeFlux << std::sqrt(std::pow(weights[0](0,0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;} */
            if(iterLoop<fastLoop){learning_rate*=factor2;}
            //if(iterLoop<fastLoop && gradientNorm>std::pow(10,-2)){learning_rate*=factor2; learning_rate=std::min(learning_rate,0.5);}
            epochForward++;
        }
        else
        {
            std::copy(weightsRetour.begin(),weightsRetour.end(),weights.begin()); std::copy(biasRetour.begin(),biasRetour.end(),bias.begin());
            cost=costPrec; gradient=gradientRetour;
            learning_rate/=factor1;
            convergence=true;
        }
        std::cout << "lr: " << learning_rate << std::endl;
        /* std::cout << "epoch: " << epoch << std::endl;
        std::cout << "gprec: " << gradientNorm.digits() << std::endl; */
        

        if(tracking)
        {
            if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
        }

        epoch++;
        

    }
    /* std::cout << "epochForward: " << epochForward << std::endl;
    std::cout << "epoch: " << epoch << std::endl;  */
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["epochForward"]=double(epochForward); study["finalGradient"]=gradientNorm; 
    study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"]=prop_entropie/epoch;}

    return study;

}

std::map<std::string,double> PGD0(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_PGD0_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PGD0_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_PGD0_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_PGD0_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, epochForward=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    std::vector<Eigen::MatrixXd> weightsRetour(L);
    std::vector<Eigen::VectorXd> biasRetour(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), gradientRetour(N), gradientInter(N);

    double V_dot, gradientNorm;
    double learning_rate = learning_rate_init;

    double prop_entropie=0, seuilE=0.01;
    double cost,costPrec,costInter;
    double lambda=0, dotProd;
    bool projection, convergence=true;
    double eps_R = std::pow(eps,2)/10;
    double const factor1=2, factor2=1.5;
    int iterLoop=0, fastLoop=10;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);
    /* double const x=-2, y=1; 
    if(record){costsFlux << cost << std::endl; normeFlux << std::sqrt(std::pow(weights[0](0,0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;} */
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot);
    //std::cout << "gprec: " << gradientNorm.digits() << std::endl;
    while (gradientNorm>eps && epoch<maxEpoch)
    {
        std::copy(weights.begin(),weights.end(),weightsRetour.begin()); std::copy(bias.begin(),bias.end(),biasRetour.begin());

        costInter = cost-learning_rate*V_dot;
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        costPrec=cost; cost = risk(Y,P,As[L],type_perte);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradientInter,type_perte);
        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());

        dotProd = learning_rate*gradient.dot(gradientInter);
        lambda=0;
        projection=(std::abs(cost-costInter)>eps_R);
        iterLoop=0;
        while(projection && iterLoop<1000)
        {
            if(std::abs(dotProd)<std::pow(10,-8)){convergence=false; break;}
            lambda -= (cost-costInter)/dotProd;
            update(L,nbNeurons,globalIndices,weights,bias,lambda*learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            cost = risk(Y,P,As[L],type_perte);
            projection=(std::abs(cost-costInter)>eps_R);
            if((iterLoop>100 && std::abs(cost-costInter)>std::pow(10,6)) || std::isnan(cost-costInter) || std::isinf(cost-costInter)){convergence=false; break;}
            if(projection){backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradientInter,type_perte); 
                dotProd = learning_rate*gradient.dot(gradientInter);
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());}
            iterLoop++;
        }

        if(convergence && !projection)
        {
            std::cout << "iterLoop: " << iterLoop << std::endl;
            std::cout << "dE: " << learning_rate*gradientNorm*gradientNorm << std::endl;
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot);
            /* if(record){costsFlux << cost << std::endl; etaFlux << learning_rate << std::endl;
            normeFlux << std::sqrt(std::pow(weights[0](0,0)-x,2)+std::pow(bias[0](0)-y,2)) << std::endl;} */
            if(iterLoop<fastLoop){learning_rate*=factor2;}
            //if(iterLoop<fastLoop && gradientNorm>std::pow(10,-2)){learning_rate*=factor2; learning_rate=std::min(learning_rate,0.5);}
            epochForward++;
        }
        else
        {
            std::copy(weightsRetour.begin(),weightsRetour.end(),weights.begin()); std::copy(biasRetour.begin(),biasRetour.end(),bias.begin());
            cost=costPrec; gradient=gradientRetour;
            learning_rate/=factor1;
            convergence=true;
        }
        std::cout << "lr: " << learning_rate << std::endl;
        /* std::cout << "epoch: " << epoch << std::endl;
        std::cout << "gprec: " << gradientNorm.digits() << std::endl; */
        

        if(tracking)
        {
            if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
        }

        epoch++;
        

    }
    /* std::cout << "epochForward: " << epochForward << std::endl;
    std::cout << "epoch: " << epoch << std::endl;  */
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["epochForward"]=double(epochForward); study["finalGradient"]=gradientNorm; 
    study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"]=prop_entropie/epoch;}

    return study;

}

std::map<std::string,double> PGRK2(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_PGRK2_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PGRK2_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_PGRK2_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_PGRK2_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, epochForward=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    std::vector<Eigen::MatrixXd> weightsRetour(L);
    std::vector<Eigen::VectorXd> biasRetour(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), gradient2(N);


    double V_dot, inv, gradientNorm;
    double learning_rate = learning_rate_init;

    double prop_entropie=0, seuilE=0.01;
    double cost,costPrec,costInter;
    double lambda=0;
    bool projection, convergence=true;
    double eps_R = std::pow(eps,2)/10;
    double const factor1=2, factor2=1.5;
    int iterLoop=0, fastLoop=5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    std::copy(weights.begin(),weights.end(),weightsRetour.begin()); std::copy(bias.begin(),bias.end(),biasRetour.begin());
    update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient2,type_perte);
    std::copy(weightsRetour.begin(),weightsRetour.end(),weights.begin()); std::copy(biasRetour.begin(),biasRetour.end(),bias.begin());
    V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot); V_dot+=gradient2.squaredNorm();
    //std::cout << "gprec: " << gradientNorm.digits() << std::endl;
    while (gradientNorm>eps && epoch<maxEpoch)
    {

        costInter = cost-0.5*learning_rate*V_dot;
        update(L,nbNeurons,globalIndices,weights,bias,-0.5*learning_rate*(gradient+gradient2));

        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        costPrec=cost; cost = risk(Y,P,As[L],type_perte);
        gradient2=gradient; backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        inv = gradient.squaredNorm();
        lambda=0;
        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        projection=(std::abs(cost-costInter)>eps_R);
        iterLoop=0;
        while(projection && iterLoop<1000)
        {
            lambda -= (cost-costInter)/inv;
            update(L,nbNeurons,globalIndices,weights,bias,lambda*gradient);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            cost = risk(Y,P,As[L],type_perte);
            projection=(std::abs(cost-costInter)>eps_R);
            if((iterLoop>20 && std::abs(cost-costInter)>std::pow(10,6)) || std::isnan(cost-costInter) || std::isinf(cost-costInter)){convergence=false; break;}
            //if((projection && cost>std::pow(10,6)) || std::isnan(cost) || std::isinf(cost) || numericalNoise(cost)){convergence=false; break;}
            if(projection){std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());}
            iterLoop++;
        }

        if(convergence && !projection)
        {
            std::cout << "iterLoop: " << iterLoop << std::endl;
            std::cout << "dE: " << learning_rate*gradientNorm*gradientNorm << std::endl;
            std::cout << "gNorm: " << gradientNorm << std::endl;
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(iterLoop<fastLoop){learning_rate*=factor2; learning_rate=std::min(learning_rate,0.5);}
            //if(iterLoop<fastLoop && gradientNorm>std::pow(10,-2)){learning_rate*=factor2; learning_rate=std::min(learning_rate,0.5);}

            epochForward++;
        }
        else
        {
            std::copy(weightsRetour.begin(),weightsRetour.end(),weights.begin()); std::copy(biasRetour.begin(),biasRetour.end(),bias.begin());
            cost=costPrec; gradient=gradient2;
            learning_rate/=factor1;
            convergence=true;
        }
        std::copy(weights.begin(),weights.end(),weightsRetour.begin()); std::copy(bias.begin(),bias.end(),biasRetour.begin());
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient2,type_perte);
        std::copy(weightsRetour.begin(),weightsRetour.end(),weights.begin()); std::copy(biasRetour.begin(),biasRetour.end(),bias.begin());
        V_dot = gradient.squaredNorm(); gradientNorm = std::sqrt(V_dot); V_dot+=gradient2.squaredNorm();

        std::cout << "lr: " << learning_rate << std::endl;
        /* std::cout << "epoch: " << epoch << std::endl;
        std::cout << "gprec: " << gradientNorm.digits() << std::endl; */
        

        if(tracking)
        {
            if((cost-costPrec)/costPrec>seuilE){prop_entropie++;}
        }

        epoch++;
        

    }
    /* std::cout << "epochForward: " << epochForward << std::endl;
    std::cout << "epoch: " << epoch << std::endl;  */
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["epochForward"]=double(epochForward); study["finalGradient"]=gradientNorm; 
    study["finalCost"]=cost; study["time"]=double(time);
    if(tracking){study["prop_entropie"]=prop_entropie/epoch;}

    return study;

}

std::map<std::string,double> PM(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, double const& learning_rate_init,
double const& beta1_init, double const& eps, int const& maxEpoch, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_PM_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_PM_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_PM_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, l;

    std::vector<Eigen::MatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::MatrixXd> slopes(L);
    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    std::vector<Eigen::MatrixXd> weightsRetour(L);
    std::vector<Eigen::VectorXd> biasRetour(L);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(N), gradientRetour(N);
    Eigen::VectorXd moment1=Eigen::VectorXd::Zero(N), moment1Prec(N), moment1Retour(N);


    double inv,vSquare=0, vSquareRetour, normV=0, gradientNorm;
    double learning_rate = learning_rate_init, beta1 = beta1_init;
    double const beta_bar = beta1/learning_rate;

    double prop_entropie=0, seuilE=0.01;
    double E,EPrec,EInter;
    double lambda=0;
    double const factor1=2, factor2=1.5;
    bool projection,convergence=true;
    double eps_E = std::pow(eps,2)/100;
    int iterLoop=0, fastLoop=5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    E = beta_bar*risk(Y,P,As[L],type_perte);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    gradientNorm = gradient.norm();
    while ((gradientNorm>eps || normV>eps) && epoch<maxEpoch)
    {
        std::copy(weights.begin(),weights.end(),weightsRetour.begin()); std::copy(bias.begin(),bias.end(),biasRetour.begin()); moment1Retour=moment1;
        vSquareRetour=vSquare;

        EInter = E-beta1*vSquare;
        update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);
        moment1 = (1-beta1)*moment1+beta1*gradient;

        vSquare=moment1.squaredNorm();
        fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        EPrec=E; E = beta_bar*risk(Y,P,As[L],type_perte)+0.5*vSquare;
        gradientRetour=gradient; backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        inv = std::pow(beta_bar,2)*gradient.squaredNorm()+vSquare;
        lambda=0;
        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin()); moment1Prec=moment1;
        projection = (std::abs(E-EInter)>eps_E);
        iterLoop=0;
        while(projection && iterLoop<1000)
        {
            lambda -= (E-EInter)/inv;
            update(L,nbNeurons,globalIndices,weights,bias,lambda*beta_bar*gradient);
            moment1*=(1+lambda); vSquare=moment1.squaredNorm();
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            E = beta_bar*risk(Y,P,As[L],type_perte)+0.5*vSquare;
            projection = (std::abs(E-EInter)>eps_E);
            if((projection && std::abs(E)>std::pow(10,6)) || std::isnan(E) || std::isinf(E)){convergence=false; break;}
            if(projection){std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin()); moment1=moment1Prec;}
            iterLoop++;
        }

        if(!projection && convergence)
        {
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            gradientNorm=gradient.norm();
            normV=std::sqrt(vSquare);
            if(iterLoop<fastLoop){learning_rate*=factor2; beta1*=factor2;}
        }
        else
        {
            std::copy(weightsRetour.begin(),weightsRetour.end(),weights.begin()); std::copy(biasRetour.begin(),biasRetour.end(),bias.begin()); moment1=moment1Retour;
            E=EPrec; gradient=gradientRetour; vSquare=vSquareRetour;
            learning_rate/=factor1; beta1/=factor1;
            convergence=true;
        }
        /* std::cout << "lr: " << learning_rate << std::endl;
        std::cout << "beta1: " << beta1 << std::endl; */

        if(tracking)
        {
            if((E-EPrec)/EPrec>seuilE){prop_entropie+=1;}
        }

        epoch+=1;

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,double> study;
    std::cout << "vnorm: " << normV << std::endl;
    study["epoch"]=double(epoch); study["finalGradient"]=gradientNorm; study["finalCost"]=(E-0.5*vSquare)/beta_bar; study["time"]=double(time);
    if(tracking){study["prop_entropie"]=prop_entropie/epoch;}

    return study;

}

std::map<std::string,double> train_Essai(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, double const& beta1, double const& beta2, double const& mu_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="PGD")
    {
        study = PGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="HolderHB")
    {
        study = HolderHB(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LC_RK")
    {
        study = LC_RK(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="RK")
    {
        study = RK(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="PGRK2")
    {
        study = PGRK2(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="PGD_Brent")
    {
        study = PGD_Brent(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="PGD0")
    {
        study = PGD0(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="PM")
    {
        study = PM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="PER")
    {
        study = PM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="LM_ER")
    {
        study = LM_ER(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,mu_init,seuil,eps,maxEpoch,tracking,track_continuous,record,fileExtension);
    }


    return study;
}