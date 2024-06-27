#include "perso.h"

std::map<std::string,Sdouble> splitting_LCEGD(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, int const& batch_size, Sdouble const& f1, Sdouble const& f2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_splitting_LCEGD"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, total_iterLoop=1, l;
    assert(batch_size<=P);
    bool condition=false;

    Sdouble lr, lr0, lr1, cost, costPrec, Vdot0, Vdot1, gradientTotNorm=1000;
    lr0=learning_rate_init; lr1=learning_rate_init;
    Sdouble const lambda=0.5;
    Sdouble alpha, beta, gamma, mu=1;
    
    Eigen::SMatrixXd echantillonY;
    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N); Eigen::SVectorXd gradientTot = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd c = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd weights_vector(N);

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    while (gradientTotNorm+std::abs(gradientTotNorm.error)>eps && iter<maxIter)
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

        echantillonY=Y.middleCols(0,batch_size);
        As[0]=X.middleCols(0,batch_size); fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        Vdot0=gradient.squaredNorm();
        costPrec=risk(echantillonY, batch_size,As[L],type_perte);

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        iterLoop=0;
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-lr0*gradient);
            fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,batch_size,As[L],type_perte);
            condition=(cost-costPrec>-lambda*lr0*Vdot0);
            if(condition)
            {
                lr0/=f1;
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        update(L,nbNeurons,globalIndices,weights,bias,lr0*c);

        echantillonY=Y.middleCols(batch_size,batch_size);
        As[0]=X.middleCols(batch_size,batch_size); fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        costPrec=risk(echantillonY,batch_size,As[L],type_perte);
        Vdot1=gradient.squaredNorm();

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-lr1*gradient);
            fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,batch_size,As[L],type_perte);
            condition=(cost-costPrec>-lambda*lr1*Vdot1);
            if(condition)
            {
                lr1/=f1;
                std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
            }
            iterLoop++;
        }while(condition);
        update(L,nbNeurons,globalIndices,weights,bias,-lr1*c);

        //lr=Sstd::max(lr0,lr1);
        lr=(lr0+lr1)/2;
        //std::cout << "lr0: " << lr0 << std::endl;
        //std::cout << "lr1: " << lr1 << std::endl;
        //std::cout << "weights: " << norme(weights,bias) << std::endl;
        //if(lr<__DBL_EPSILON__){break;}
        if(numericalNoise(Vdot0) || numericalNoise(Vdot1)){break;}

        std::cout << "c: " << c.norm() << std::endl;
        tabToVector(weights,bias,L,nbNeurons,globalIndices,weights_vector); c+=1/(2*lr)*weights_vector;
        tabToVector(weightsPrec,biasPrec,L,nbNeurons,globalIndices,weights_vector); c+=1/(2*lr)*weights_vector;
        tabToVector(weightsInter,biasInter,L,nbNeurons,globalIndices,weights_vector); c-=1/lr*weights_vector;

        /* tabToVector(weights,bias,L,nbNeurons,globalIndices,weights_vector); alpha = 1/(2*lr1); c+=alpha*weights_vector;
        tabToVector(weightsPrec,biasPrec,L,nbNeurons,globalIndices,weights_vector); beta = 1/(2*lr0); c+=beta*weights_vector;
        tabToVector(weightsInter,biasInter,L,nbNeurons,globalIndices,weights_vector); gamma=-alpha-beta; c+=gamma*weights_vector; */

        if(lr<__DBL_EPSILON__){break;}

        //lr0*=f2; lr1*=f2;
        //lr0=Sstd::min(lr0,lr1)*f2; lr1=Sstd::min(lr0,lr1)*f2;
        //lr0=lr*f2; lr1=lr*f2;
        //lr0=Sstd::min(lr0,1); lr1=Sstd::min(lr1,1);
        //lr0=1; lr1=1;

        lr0*=f2; lr1*=f2;
        if(c.norm()<__DBL_EPSILON__){}
        else{lr0=Sstd::min(lr0, mu/c.norm()); lr1=Sstd::min(lr1, mu/c.norm());}

        As[0]=X; fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradientTot,type_perte);
        gradientTotNorm = gradientTot.norm();
        total_iterLoop+=iterLoop;

        iter++;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,Sdouble> study;

    As[0]=X;
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte);

    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(), study["finalCost"]=cost; study["time"]=Sdouble(time);
    study["total_iterLoop"]=Sdouble(total_iterLoop);

    return study;

}

std::map<std::string,Sdouble> splitting2_LCEGD(Eigen::SMatrixXd const& X, Eigen::SMatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte,
Sdouble const& learning_rate_init, int const& batch_size, Sdouble const& f1, Sdouble const& f2, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_splitting2_LCEGD"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, total_iterLoop=1, l;
    assert(batch_size<=P);
    bool condition=false;

    Sdouble eta=learning_rate_init, eta_prec=0, mu=learning_rate_init, mu_prec=0, cost, costPrec, gradientTotNorm=1000;
    Sdouble sum_cost=0, sum_costPrec=0;
    Sdouble coeff;
    
    Eigen::SMatrixXd echantillonY;
    std::vector<Eigen::SMatrixXd> As(L+1);
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd grad0 = Eigen::SVectorXd::Zero(N); Eigen::SVectorXd grad1 = Eigen::SVectorXd::Zero(N); 
    Eigen::SVectorXd gradientTot = Eigen::SVectorXd::Zero(N);

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    while (gradientTotNorm+std::abs(gradientTotNorm.error)>eps && iter<maxIter)
    {
        sum_cost=0; sum_costPrec=0;
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

        echantillonY=Y.middleCols(0,batch_size);
        As[0]=X.middleCols(0,batch_size); fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad0,type_perte);
        if(mu_prec>__DBL_EPSILON__ || eta_prec<__DBL_EPSILON__)
        {
            condition=(grad0.squaredNorm()+grad0.dot(grad1)>=0);
            if(condition)
            {
                costPrec=risk(echantillonY,batch_size,As[L],type_perte);
                std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
                iterLoop=0;
                do
                {
                    if(eta_prec<__DBL_EPSILON__  && mu_prec<__DBL_EPSILON__){update(L,nbNeurons,globalIndices,weights,bias,-0.5*eta*(grad0+grad1));}
                    else
                    {   
                        coeff=mu_prec/(eta_prec+mu_prec);
                        update(L,nbNeurons,globalIndices,weights,bias,-coeff*eta*(grad0+grad1));
                    }
                    fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,batch_size,As[L],type_perte);
                    condition=(cost-costPrec>0);
                    if(condition)
                    {
                        eta/=f1;
                        std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    }
                    iterLoop++;
                }while(condition);
                sum_cost+=cost; sum_costPrec+=costPrec;
            }
            else{eta=0;}
        }
        else{eta=0;}

        echantillonY=Y.middleCols(batch_size,batch_size);
        As[0]=X.middleCols(batch_size,batch_size); fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,batch_size,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad1,type_perte);
        if(eta_prec>__DBL_EPSILON__ || mu_prec<__DBL_EPSILON__)
        {
            condition=(grad1.squaredNorm()+grad1.dot(grad0)>=0);
            if(condition)
            {
                costPrec=risk(echantillonY,batch_size,As[L],type_perte);
                std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
                do
                {
                    if(eta_prec<__DBL_EPSILON__  && mu_prec<__DBL_EPSILON__ ){update(L,nbNeurons,globalIndices,weights,bias,-0.5*mu*(grad0+grad1));}
                    else
                    {
                        coeff=eta_prec/(eta_prec+mu_prec);
                        update(L,nbNeurons,globalIndices,weights,bias,-coeff*mu*(grad0+grad1));
                    }
                    fforward(L,batch_size,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,batch_size,As[L],type_perte);
                    condition=(cost-costPrec>0);
                    if(condition)
                    {
                        mu/=f1;
                        std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    }
                    iterLoop++;
                }while(condition);
                sum_cost+=cost; sum_costPrec+=costPrec;
            }
            else{mu=0;}
        }
        else{mu=0;}

        eta_prec=eta; mu_prec=mu;
        eta=learning_rate_init; mu=learning_rate_init;

        As[0]=X; fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradientTot,type_perte);
        gradientTotNorm = gradientTot.norm();
        if(numericalNoise(gradientTotNorm)){break;}
        total_iterLoop+=iterLoop;

        iter++;
        //if(sum_cost>sum_costPrec){std::cout << "bug! " << "iter: " << iter << std::endl; std::cout << sum_cost-sum_costPrec << std::endl;}
        //std::cout << sum_cost-sum_costPrec << std::endl;
        std::cout << grad1.norm() << std::endl;
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,Sdouble> study;

    cost = risk(Y,P,As[L],type_perte);

    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientTot.norm(), study["finalCost"]=cost; study["time"]=Sdouble(time);
    study["total_iterLoop"]=Sdouble(total_iterLoop);

    return study;

}

std::map<std::string,Sdouble> LC_Mechanic(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_Mechanic_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_Mechanic_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N), moment1 = Eigen::SVectorXd::Zero(N), moment1Prec;

    Sdouble gradientNorm;
    Sdouble Em_count=0;
    Sdouble cost, E, EPrec, E0, gE, gE0, vsquarePrec=0, vsquare;
    bool condition;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble const seuilE=0.0, lambda=0.5;
    Sdouble learning_rate;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); E=cost; E0=E; if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm(); learning_rate = learning_rate_init; gE0 = gradientNorm;

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //Sdouble x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while(gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        moment1Prec = moment1; vsquarePrec=vsquare;
        EPrec=E; gE = Sstd::sqrt(gradientNorm*gradientNorm+vsquare);
        learning_rate = Sstd::min(learning_rate,Sstd::sqrt(E0/E));

        do
        {   
            moment1 = (1-learning_rate*Sstd::sqrt(E/E0))*moment1-learning_rate*gradient;
            update(L,nbNeurons,globalIndices,weights,bias,learning_rate*moment1);
            vsquare = moment1.squaredNorm();
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte); E = cost+0.5*vsquare;
            condition=(E-EPrec>-lambda*learning_rate*Sstd::sqrt(E/E0)*vsquare);
            //condition=(Sstd::sqrt(E)-Sstd::sqrt(EPrec)>-0.5*lambda*learning_rate*vsquare/Sstd::sqrt(E0));
            if(condition)
            {
                //learning_rate = rho*learning_rate*(-(lambda-1)*beta_bar*vsquare)/(Sstd::max((E-EPrec)/learning_rate+beta_bar*vsquare,-eps0*(lambda-1)*beta_bar*vsquare));
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
        //learning_rate=Sstd::sqrt(E0/E);
        //learning_rate=Sstd::min(10000*learning_rate,Sstd::sqrt(E0/E));
        //std::cout << "lrAprès: " << learning_rate << std::endl;
        //learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(Sstd::max((E-EPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));
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

            //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();

        if(E-EPrec>0)
        {
            Em_count+=1;
        }

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> LC_M(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_M_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_M_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N), moment1 = Eigen::SVectorXd::Zero(N), moment1Prec;

    Sdouble gradientNorm;
    Sdouble const beta_bar=10;
    Sdouble Em_count=0;
    Sdouble cost, E, EPrec, vsquare;
    bool condition;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble const seuilE=0.0, learning_rate_max=1/beta_bar, lambda=0.01;
    Sdouble const rho=0.9, eps0=std::pow(10,-2);
    Sdouble learning_rate = std::min(learning_rate_init,learning_rate_max);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); E=cost; if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm();

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //Sdouble x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while(gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        moment1Prec = moment1;
        EPrec=E;

        learning_rate=std::min(learning_rate,learning_rate_max);
        //learning_rate = learning_rate_max;
        do
        {   
            moment1 = (1-beta_bar*learning_rate)*moment1+learning_rate*gradient;
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*moment1);
            vsquare = moment1.squaredNorm();
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte); E = cost+0.5*vsquare;
            condition=(E-EPrec>-beta_bar*lambda*learning_rate*vsquare);
            if(condition)
            {
                //learning_rate = rho*learning_rate*(-(lambda-1)*beta_bar*vsquare)/(Sstd::max((E-EPrec)/learning_rate+beta_bar*vsquare,-eps0*(lambda-1)*beta_bar*vsquare));
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
        //std::cout << "lrAprès: " << learning_rate << std::endl;
        //learning_rate = rho*learning_rate*(-(lambda-1)*V_dot)/(Sstd::max((E-EPrec)/learning_rate+V_dot,-eps0*(lambda-1)*V_dot));
        //std::cout << learning_rate << std::endl;

        if(record)
        {
            for(l=0;l<L;l++)
            {
                weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
                weightsFlux << weights[l] << std::endl;
                weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
                weightsFlux << bias[l] << std::endl;
            }

            //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();

        if(E-EPrec>0)
        {
            Em_count+=1;
        }

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> LC_clipping(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_clipping_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_clipping_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm;
    Sdouble Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec;
    bool condition;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble seuilE=0.0;
    Sdouble const gamma=10*eps, lambda=0.5;
    Sdouble learning_rate = learning_rate_init;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm();

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //Sdouble x=-4, y=3;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;

        do
        {
            update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*Sstd::min(1,gamma/gradientNorm)*gradient);
            //update(L,nbNeurons,globalIndices,weights,bias,-learning_rate*gradient/gradientNorm);
            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost = risk(Y,P,As[L],type_perte);
            condition=(cost-costPrec>-lambda*learning_rate*Sstd::min(1,gamma/gradientNorm)*Sstd::pow(gradientNorm,2));
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

            //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> LC_signGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_signGD_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_signGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm;
    Sdouble Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec;
    bool condition;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble seuilE=0.0;
    Sdouble const lambda=0.5;
    Sdouble learning_rate = learning_rate_init;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    gradientNorm=gradient.norm();

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //Sdouble x=-4, y=3;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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

            //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> LC_EM(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, 
Sdouble const& learning_rate_init, Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LC_Em_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_LC_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, l;
    Sdouble beta_bar = beta1_init/learning_rate_init;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble vsquare, prod, cost, costPrec;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble learning_rate = learning_rate_init;
    Sdouble const seuil=0.0, f1=2, f2=10000;
    Sdouble a,b,c, delta, x1,x2,x;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costPrec=cost;
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
                a = (2+beta_bar*learning_rate)*(4*vsquare+Sstd::pow(beta_bar*learning_rate*gradientNorm,2)-4*beta_bar*learning_rate*prod);
                b = 2*beta_bar*learning_rate*prod-4*vsquare;
                c = beta_bar*(cost-costPrec);
                delta = b*b-4*a*c;
                if(delta<0 || Sstd::abs(a)<std::pow(10,-12))
                {
                    learning_rate/=f1; 
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                }
            }while(delta<0 || Sstd::abs(a)<std::pow(10,-12));
            x1=(-b-Sstd::sqrt(delta))/(2*a); x2=(-b+Sstd::sqrt(delta))/(2*a);
            if(Sstd::abs(x1)<Sstd::abs(x2)){x=x2;}
            else{x=x1;}
            moment1 = (4*x-1)*moment1-2*beta_bar*learning_rate*x*gradient;
        }
        std::cout << "eta_avant: " << learning_rate << std::endl;
        learning_rate*=f2;
        //std::cout << "learning_rate: " << learning_rate << std::endl;

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm=gradient.norm();
        std::cout << "iter: " << iter << std::endl;
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> LC_EGD2(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_EGD2_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_EGD2_"+fileExtension+".csv").c_str());
    //std::ofstream evalFlux(("Record/eval_LC_EGD2_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, total_iterLoop=1, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm;
    Sdouble Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec, V_dot;
    bool condition;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble seuilE=0.0;
    Sdouble f1=30, f2=10000, rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    Sdouble gauche, droite, m, m_best;
    int const nLoops=3; bool last_pass=false; bool active_other=false;
    Sdouble learning_rate = learning_rate_init;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    V_dot=gradient.squaredNorm(); gradientNorm=Sstd::sqrt(V_dot);

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    Sdouble x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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
            gauche = Sstd::log10(learning_rate); droite = Sstd::log10(f1*learning_rate);
            for (int k=0; k<nLoops; k++)
            {
                m=(gauche+droite)/2; learning_rate=Sstd::pow(10,m);
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
                        learning_rate=Sstd::pow(10,m_best);
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

            normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot=gradient.squaredNorm(); gradientNorm=Sstd::sqrt(V_dot);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        //std::cout << "iter: "<< iter << " and cost: "<< cost << std::endl;
        //std::cout << "gradnorm: " << gradientNorm << std::endl;

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;

    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,0.9,0.999,eps,maxIter,tracking,record,fileExtension);
    }

    if(!active_other){
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=Sdouble(time); study["total_iterLoop"]=Sdouble(total_iterLoop);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["iter"]+Em_count)/(study["iter"]+Sdouble(iter));}
        study["iter"]+=Sdouble(iter); study["time"]+=Sdouble(time); study["total_iterLoop"]=Sdouble(total_iterLoop);
    }

    return study;

}

std::map<std::string,Sdouble> LC_EGD3(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_EGD3_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_EGD3_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm;
    Sdouble Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec, V_dot;
    bool condition;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble seuilE=0.0;
    Sdouble const rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    Sdouble learning_rate = learning_rate_init;
    bool active_other=false;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    V_dot=gradient.squaredNorm(); gradientNorm=Sstd::sqrt(V_dot);

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    //Sdouble x=-4, y=3;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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

        if(isnan(gradientNorm) or numericalNoise(gradientNorm) or learning_rate<std::pow(10,-10)){active_other=true; break;}

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

            //normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot=gradient.squaredNorm(); gradientNorm=Sstd::sqrt(V_dot);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        iter++;

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    std::map<std::string,Sdouble> study;
    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,1-0.9,1-0.999,eps,maxIter,tracking,record,fileExtension);
    }

    if(!active_other){
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["iter"]+Em_count)/(study["iter"]+Sdouble(iter));}
        study["iter"]+=Sdouble(iter); study["time"]+=Sdouble(time);
    }

    return study;

}

std::map<std::string,Sdouble> LC_EGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LC_EGD_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LC_EGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, total_iterLoop=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm;
    Sdouble Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec, V_dot;
    bool condition, active_other=false;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble seuilE=0.0, learning_rate_max=1000, rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    Sdouble learning_rate = std::min(learning_rate_init,learning_rate_max);
    Sdouble const f1=2, f2=10000;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); if(record){costsFlux << cost << std::endl;}
    V_dot=gradient.squaredNorm(); gradientNorm=Sstd::sqrt(V_dot);

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    Sdouble x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while(gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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

            normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        V_dot=gradient.squaredNorm(); gradientNorm=Sstd::sqrt(V_dot);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;

    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,1-0.9,1-0.999,eps,maxIter,tracking,record,fileExtension);
    }

    if(!active_other){
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=Sdouble(time); study["total_iterLoop"]=Sdouble(total_iterLoop);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["iter"]+Em_count)/(study["iter"]+Sdouble(iter));}
        study["iter"]+=Sdouble(iter); study["time"]+=Sdouble(time); study["total_iterLoop"]=Sdouble(total_iterLoop);
    }

    return study;

}

std::map<std::string,Sdouble> LCI_EGD(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter, bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream gradientNormFlux(("Record/gradientNorm_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream normeFlux(("Record/norme_LCI_EGD_"+fileExtension+".csv").c_str());
    std::ofstream etaFlux(("Record/eta_LCI_EGD_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterLoop=0, total_iterLoop=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm;
    Sdouble Em_count=0,continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec, costInit, grad_square, V_dot=0;
    bool condition, active_other=false;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble seuilE=0.0, learning_rate_max=1000, rho=0.9, eps0=std::pow(10,-2), lambda=0.5;
    Sdouble learning_rate = std::min(learning_rate_init,learning_rate_max);
    Sdouble const f1=2, f2=10000;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit=cost; if(record){costsFlux << cost << std::endl;}
    grad_square = gradient.squaredNorm(); V_dot=grad_square; gradientNorm=Sstd::sqrt(grad_square);

    //std::cout << "cInit: " << cost << std::endl;
    //std::cout << "grInit: " << gradientNorm << std::endl;

    Sdouble x=0, y=-1;
    if(record)
    {
        for(l=0;l<L;l++)
        {
            weights[l].resize(nbNeurons[l+1]*nbNeurons[l],1);
            weightsFlux << weights[l] << std::endl;
            weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
            weightsFlux << bias[l] << std::endl;
        }

        normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
    }

    while(gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
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

            normeFlux << Sstd::sqrt(Sstd::pow(weights[0](0)-x,2)+Sstd::pow(bias[0](0)-y,2)) << std::endl;
        }

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        grad_square = gradient.squaredNorm(); V_dot+=grad_square; gradientNorm=Sstd::sqrt(grad_square);

        if(cost-costPrec>0)
        {
            Em_count+=1;
        }

        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;

    if(active_other)
    {
        study = Adam(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,0.001,P,1-0.9,1-0.999,eps,maxIter,tracking,record,fileExtension);
    }

    if(!active_other){
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm, study["finalCost"]=cost; study["time"]=Sdouble(time); study["total_iterLoop"]=Sdouble(total_iterLoop);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter);}
    }
    else
    {   
        if(tracking){study["prop_entropie"] = (study["prop_entropie"]*study["iter"]+Em_count)/(study["iter"]+Sdouble(iter));}
        study["iter"]+=Sdouble(iter); study["time"]+=Sdouble(time); study["total_iterLoop"]=Sdouble(total_iterLoop);
    }

    return study;

}

std::map<std::string,Sdouble> EulerRichardson(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream gradientNormFlux(("Record/gradientNorm_EulerRichardson_"+fileExtension+".csv").c_str());
    std::ofstream costsFlux(("Record/cost_EulerRichardson_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_EulerRichardson_"+fileExtension+".csv").c_str());
    if(!gradientNormFlux || !costsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd gradientInter = Eigen::SVectorXd::Zero(N);


    Sdouble gradientNorm = 1000;
    Sdouble learning_rate = learning_rate_init;
    Sdouble erreur;

    Sdouble prop_entropie=0, prop_initial_ineq=0, modif=0, seuilE=0.01;
    Sdouble costInit,cost,costPrec;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    while (gradientNorm+std::abs(gradientNorm.error)>eps && iter<maxIter)
    {
        //std::cout << "cost: " << cost << std::endl;

       if(iter==0)
        {
            std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
            if(tracking){cost = risk(Y,P,As[L],type_perte); costInit = cost;}

            if(record)
            {
                if(tracking){costsFlux << cost.number << std::endl;}
                gradientNorm = gradient.norm(); gradientNormFlux << gradientNorm.number << std::endl;
            }

            update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
            fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
            backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        }

        erreur=(0.5*learning_rate*(gradient-gradientInter).norm())/seuil;

        if(erreur>1)
        {
           learning_rate*=0.9/Sstd::sqrt(erreur);
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
                if(!std::signbit((cost-costInit).number)){prop_initial_ineq++;}

                if(record){speedFlux << ((cost-costPrec)/learning_rate + gradient.squaredNorm()).number << std::endl;}
            }
            backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

            //std::cout << learning_rate << std::endl;
            learning_rate*=0.9/Sstd::sqrt(erreur);
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
        update(L,nbNeurons,globalIndices,weightsInter,biasInter,-learning_rate*gradient);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,type_perte);

        gradientNorm = gradientInter.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;

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

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}

//Diminution naive du pas pour GD
std::map<std::string,Sdouble> GD_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_GD_Em_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_GD_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterTot=0, l;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble cost,costPrec,costInit;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble learning_rate = 0.1;
    Sdouble seuilE=0.0, facteur=1.5, factor2=1.5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
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

        std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        costPrec=cost;
        learning_rate=0.1;
        do
        {
            //iterTot++;
            //if(iterTot<10){std::cout << learning_rate << std::endl;}
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
        if(record){speedFlux << ((cost-costPrec)/learning_rate + gradient.squaredNorm()).number << std::endl;}

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
        gradientNorm=gradient.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> LM_ER(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& mu_init,
Sdouble const& seuil, Sdouble const& eps, int const& maxIter, bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_LM_ER_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, l;

    Sdouble prop_entropie=0, prop_initial_ineq=0, modif=0;
    Sdouble costInit, cost, costPrec;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);

     std::vector<Eigen::SMatrixXd> weightsInter(L);
    std::vector<Eigen::SVectorXd> biasInter(L);

    Eigen::SVectorXd gradient(N), gradientInter(N), delta(N), deltaInter(N);
    Eigen::SMatrixXd Q(N,N), QInter(N,N);
    Eigen::SMatrixXd I = Eigen::SMatrixXd::Identity(N,N);

    Sdouble gradientNorm = 1000;
    Sdouble h = 1/mu_init;
    Sdouble mu = mu_init;
    Sdouble erreur;

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
           h*=0.9/Sstd::sqrt(erreur);
        }
        else
        {
            update(L,nbNeurons,globalIndices,weights,bias,h*deltaInter);
            h*=0.9/Sstd::sqrt(erreur);

            fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
            QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,Q,type_perte);
            solve(gradient,Q+mu*I,delta);

            if(tracking)
            {
                modif++;
                costPrec = cost;
                cost = risk(Y,P,As[L],type_perte);
                if(std::signbit((cost-costPrec).number)){prop_entropie++;}
                if(std::signbit((cost-costInit).number)){prop_initial_ineq++;}
            }
        }

        std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());

        update(L,nbNeurons,globalIndices,weightsInter,biasInter,h*delta);
        fforward(L,P,nbNeurons,activations,weightsInter,biasInter,As,slopes);
        QSO_backward(Y,L,P,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gradientInter,QInter,type_perte);
        solve(gradientInter,QInter+mu*I,deltaInter);


        gradientNorm = gradientInter.norm();

        if(numericalNoise(gradientNorm)){break;}

        iter++;
    }

    std::copy(weightsInter.begin(),weightsInter.end(),weights.begin()); std::copy(biasInter.begin(),biasInter.end(),bias.begin());
    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    cost = risk(Y,P,As[L],type_perte);

    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradientNorm; study["finalCost"]=cost;
    if(tracking){study["prop_entropie"]=prop_entropie/modif; study["prop_initial_ineq"]=prop_initial_ineq/modif;}

    return study;

}




//Diminution naive du pas
std::map<std::string,Sdouble> Momentum_Em(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, Sdouble const& learning_rate_init,
Sdouble const& beta1_init, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{

    std::ofstream weightsFlux(("Record/weights_Momentum_Em_"+fileExtension+".csv").c_str());
    std::ofstream speedFlux(("Record/speed_Momentum_Em_"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), iter=0, iterTot=0, l;
    Sdouble beta_bar = beta1_init/learning_rate_init;

    std::vector<Eigen::SMatrixXd> As(L+1); As[0]=X;
    std::vector<Eigen::SMatrixXd> slopes(L);
    Eigen::SVectorXd gradient = Eigen::SVectorXd::Zero(N);
    Eigen::SVectorXd moment1 = Eigen::SVectorXd::Zero(N), moment1Prec(N);

    Sdouble gradientNorm=1000;
    Sdouble Em_count=0,condition, continuous_entropie=0, prop_initial_ineq=0;
    Sdouble EmPrec,Em,cost,costInit;

    std::vector<Eigen::SMatrixXd> weightsPrec(L);
    std::vector<Eigen::SVectorXd> biasPrec(L);
    Sdouble learning_rate = 0.1, beta1 = beta1_init;
    Sdouble seuil=0.0, facteur=1.5;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    fforward(L,P,nbNeurons,activations,weights,bias,As,slopes);
    backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);
    cost = risk(Y,P,As[L],type_perte); costInit = cost;
    Em = beta_bar*cost;
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

        moment1Prec=moment1; std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
        EmPrec=Em;
        learning_rate=0.1; beta1=beta1_init;
        do
        {
            //iterTot++;
            //if(iterTot<10){std::cout << learning_rate << std::endl;}
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
        if(record){speedFlux << ((Em-EmPrec)/learning_rate+beta_bar*moment1.squaredNorm()).number << std::endl;}

        backward(Y,L,P,nbNeurons,activations,globalIndices,weights,bias,As,slopes,gradient,type_perte);

        gradientNorm=gradient.norm();
        //std::cout << "gradientNorm: " << gradientNorm << std::endl;
        iter++;
        if(numericalNoise(gradientNorm)){break;}

    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    //std::cout << "gradientNorm" << gradientNorm << std::endl;
    std::map<std::string,Sdouble> study;
    study["iter"]=Sdouble(iter); study["finalGradient"]=gradient.norm(); study["finalCost"]=cost; study["time"]=Sdouble(time);
    if(tracking){study["prop_entropie"] = Em_count/Sdouble(iter); study["prop_initial_ineq"] = prop_initial_ineq/Sdouble(iter);}
    if(track_continuous){study["continuous_entropie"] = continuous_entropie/Sdouble(iter);}

    return study;

}

std::map<std::string,Sdouble> train_Perso(Eigen::SMatrixXd& X, Eigen::SMatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::SMatrixXd>& weights, std::vector<Eigen::SVectorXd>& bias, std::string const& type_perte, std::string const& algo,
Sdouble const& learning_rate_init, Sdouble const& beta1, Sdouble const& beta2, int const& batch_size, Sdouble const& mu_init, Sdouble const& seuil, Sdouble const& eps, int const& maxIter,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,Sdouble> study;

    if(algo=="EulerRichardson")
    {
        study = EulerRichardson(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,seuil,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="splitting_LCEGD")
    {
        study = splitting_LCEGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,2,10000,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="splitting2_LCEGD")
    {
        study = splitting2_LCEGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,2,10000,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_EGD")
    {
        study = LC_EGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LCI_EGD")
    {
        study = LCI_EGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_EGD2")
    {
        study = LC_EGD2(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_M")
    {
        study = LC_M(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_Mechanic")
    {
        study = LC_Mechanic(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_clipping")
    {
        study = LC_clipping(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_signGD")
    {
        study = LC_signGD(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_EGD3")
    {
        study = LC_EGD3(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,record,fileExtension);
    }
    else if(algo=="LC_EM")
    {
        study = LC_EM(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="GD_Em")
    {
        study = GD_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }
    else if(algo=="Momentum_Em")
    {
        study = Momentum_Em(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,beta1,eps,maxIter,tracking,track_continuous,record,fileExtension);
    }


    return study;
}
