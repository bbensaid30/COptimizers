#include "incremental.h"

std::map<std::string,double> RAG(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_RAG"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_iterLoop=0, l;
    assert(batch_size<=P);
    int m, taille;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}
    bool condition;

    double eta, eta0, eta1, eta_start=learning_rate_init, cost, costPrec;
    double const lamb=0.5;
    double LMax, LSum, dist=0, grad_square, gNorm=1000, prod;
    double R=0, R_epoch;
    int imax;

    double gauche, droite, milieu, m_best;
    int const nLoops=2; bool last_pass=false;
    
    double *LTab = new double[m], *diffs = new double[m], *R_tab = new double[m];
    std::vector<int> permut(m);
    std::vector<Eigen::VectorXd> grads(m);

    double const coeff_max=2*m-1, heuris_max=2;
    double heuris=1;
    double coeff=coeff_max;

    Eigen::MatrixXd echantillonX, echantillonY;
    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd grad(N), g=Eigen::VectorXd::Zero(N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //initialization
    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    for(int i=0; i<m; i++)
    {   
        permut[i]=i;

        taille = selection_data(i,m,batch_size,P,X,Y,echantillonX,echantillonY); As[0]=echantillonX;
        fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad,type_perte,false);
        costPrec = risk(echantillonY,taille,As[L],type_perte,false); total_iterLoop+=1;
        R_tab[i]=costPrec; R+=costPrec; diffs[i]=0;

        g+=grad; grads[i]=grad;
        grad_square=grad.squaredNorm();

        eta=learning_rate_init;
        condition=(grad_square>eps*eps);
        while(condition)
        {
            update(L,nbNeurons,globalIndices,weights,bias,-eta*grad);
            fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
            condition=(cost-costPrec>-lamb*eta*grad_square);
            if(condition){eta/=f1;}
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            total_iterLoop+=1;
        }

        if(grad_square<eps*eps || eta<__DBL_EPSILON__){LTab[i]=0;}
        else{LTab[i]=(2*(1-lamb))/eta;}
    }
    gNorm=g.norm(); 
    //std::cout << "grNorm: " << gNorm/P << std::endl;
    if(gNorm/P>eps)
    {
        LSum=sum_max_tab(LTab,m,LMax);
        if(LSum<__DBL_EPSILON__){eta=learning_rate_init;}
        else{eta=(2*(1-lamb))/LSum;}
        update(L,nbNeurons,globalIndices,weights,bias,-eta*g);
    }

    //general epoch
    while((gNorm/P>eps || dist/P>eps) and epoch<=maxEpoch)
    {
        R_epoch=R; 
        /* std::sort(permut.begin(), permut.end(),
        [&](const int& a, const int& b) {
            return (LTab[a] < LTab[b]);
        }
        ); */
        for(int i=0; i<m; i++)
        {
            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());

            taille = selection_data(permut[i],m,batch_size,P,X,Y,echantillonX,echantillonY); As[0]=echantillonX;
            fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes);
            backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad,type_perte,false);
            costPrec = risk(echantillonY,taille,As[L],type_perte,false); total_iterLoop+=1;

            R-=R_tab[permut[i]]; R+=costPrec; R_tab[permut[i]]=costPrec;
            g-=grads[permut[i]]; g+=grad; grads[permut[i]]=grad;
            grad_square=grad.squaredNorm(); gNorm=g.norm(); prod=g.dot(grad);

            eta0=eta_start; eta1=eta_start;
            condition=(grad_square>eps*eps);
            iterLoop=0;
            while(condition)
            {
                update(L,nbNeurons,globalIndices,weights,bias,-eta0*grad);
                fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
                condition=(cost-costPrec>-lamb*eta0*grad_square);
                if(condition){eta0/=f1;}
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                iterLoop+=1;
            }
            if(iterLoop>1 && eta0>__DBL_EPSILON__)
            {
                gauche = std::log10(eta0); droite = std::log10(f1*eta0);
                for (int k=0; k<nLoops; k++)
                {
                    milieu=(gauche+droite)/2; eta0=std::pow(10,milieu);
                    update(L,nbNeurons,globalIndices,weights,bias,-eta0*grad);
                    fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,taille,As[L],type_perte,false);
                    if(cost-costPrec>-lamb*eta0*grad_square)
                    {
                        m_best=gauche;
                        droite=milieu;
                        last_pass=false;
                    }
                    else
                    {
                        gauche=milieu;
                        last_pass=true;
                    }
                    iterLoop++;
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                }
                if(!last_pass){eta0=std::pow(10,m_best);}
            }
            total_iterLoop+=iterLoop;
            
            imax=indice_max_tab(LTab,m);

            if(prod>__DBL_EPSILON__  && permut[i]==imax && prod<gNorm*gNorm)
            {   
                condition=true;
                iterLoop=0;
                while(condition)
                {
                    update(L,nbNeurons,globalIndices,weights,bias,-eta1*g);
                    fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
                    condition=(cost-costPrec>-lamb*eta1*prod);
                    if(condition){eta1/=f1;}
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    iterLoop+=1;
                }
                if(iterLoop>1 && eta1>__DBL_EPSILON__)
                {
                    gauche = std::log10(eta1); droite = std::log10(f1*eta1);
                    for (int k=0; k<nLoops; k++)
                    {
                        milieu=(gauche+droite)/2; eta1=std::pow(10,milieu);
                        update(L,nbNeurons,globalIndices,weights,bias,-eta1*g);
                        fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,taille,As[L],type_perte,false);
                        if(cost-costPrec>-lamb*eta1*prod)
                        {
                            m_best=gauche;
                            droite=milieu;
                            last_pass=false;
                        }
                        else
                        {
                            gauche=milieu;
                            last_pass=true;
                        }
                        iterLoop++;
                        std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    }
                    if(!last_pass){eta1=std::pow(10,m_best);}
                }
                total_iterLoop+=iterLoop;
                eta=max(eta0,eta1);
            }
            else{eta=eta0;}
            
            if(grad_square<std::pow(eps,2) || eta<__DBL_EPSILON__){LTab[permut[i]]=0;}
            else{LTab[permut[i]]=(2*(1-lamb))/eta;}
            LSum=sum_max_tab(LTab,m,LMax);

            if(LSum<__DBL_EPSILON__){eta=learning_rate_init; eta_start=learning_rate_init;}
            else
            {   eta_start=f2*(2*(1-lamb))/LMax;
                if(coeff==coeff_max){eta=2/(coeff*LSum);}
                else{eta=2/(heuris*coeff*LSum);}
            }

            update(L,nbNeurons,globalIndices,weights,bias,-eta*g);
            
            dist-=diffs[permut[i]];
            if(grad_square>eps*eps && eta0>__DBL_EPSILON__){diffs[permut[i]]=(2*(1-lamb)/eta0)*eta*gNorm;}
            else{diffs[permut[i]]=0;}
            dist+=diffs[permut[i]];
            
            if(gNorm/P<eps && dist/P<eps){break;}
            permut[i]=i;
        }

        if(dist<gNorm){heuris=(heuris+heuris_max)/2;}
        else{heuris=(1+heuris)/2;}
        if(R-R_epoch<0){coeff/=heuris;}
        else{coeff=coeff_max;}

        epoch++;
        /* std::cout << "gNorm: " << gNorm/P << std::endl;
        std::cout << "dist: " << dist/P << std::endl;
        std::cout << "R: " << R/P << std::endl;
        std::cout << "coeff: " << coeff << std::endl;
        std::cout << "heuris: " << heuris << std::endl; */
    }

    delete [] LTab; delete [] diffs; delete [] R_tab;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    As[0]=X; fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost=risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gNorm/P; study["finalCost"]=cost; study["time"]=double(time);
    study["total_iterLoop"]=double(total_iterLoop);

    return study;
}

// RAGL with memory
std::map<std::string,double> RAG_L(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_RAG_L"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_iterLoop=0, l;
    assert(batch_size<=P);
    int m, taille;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}
    bool condition=false;

    double eta, eta0, eta1, eta_start=learning_rate_init, cost, costPrec;
    double const lamb=0.5;
    double LSum, LMax, LMax_now, dist=0, grad_square=0, gNorm=1000, prod;
    double R=0, R_epoch;
    int imax, imax_now;

    double gauche, droite, milieu, m_best;
    int const nLoops=2; bool last_pass=false;

    double const coeff_max=4*m-1, heuris_max=2;
    double coeff = coeff_max, heuris=1;
    
    double *LTab = new double[m], *R_tab = new double[m];
    std::vector<int> permut(m);
    Eigen::VectorXd gSum(N), gs(N);

    Eigen::MatrixXd echantillonX, echantillonY;
    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd grad(N), g=Eigen::VectorXd::Zero(N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);
    std::vector<Eigen::MatrixXd> weightsInter(L);
    std::vector<Eigen::VectorXd> biasInter(L);
    std::vector<Eigen::MatrixXd> weightsMemory(L);
    std::vector<Eigen::VectorXd> biasMemory(L);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //initialization
    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    for(int i=0; i<m; i++)
    {   
        permut[i]=i;

        taille = selection_data(i,m,batch_size,P,X,Y,echantillonX,echantillonY); As[0]=echantillonX;
        fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad,type_perte,false);
        costPrec = risk(echantillonY,taille,As[L],type_perte,false); total_iterLoop+=1;
        R_tab[i]=costPrec; R+=costPrec;

        g+=grad;
        grad_square=grad.squaredNorm();

        eta=learning_rate_init;
        condition=(grad_square>eps*eps);
        while(condition)
        {
            update(L,nbNeurons,globalIndices,weights,bias,-eta*grad);
            fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
            condition=(cost-costPrec>-lamb*eta*grad_square);
            if(condition){eta/=f1;}
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            total_iterLoop+=1;
        }
        
        if(grad_square<std::pow(eps,2) || eta<__DBL_EPSILON__){LTab[i]=0;}
        else{LTab[i]=(2*(1-lamb))/eta;}
    }
    gNorm=g.norm();
    std::copy(weights.begin(),weights.end(),weightsInter.begin()); std::copy(bias.begin(),bias.end(),biasInter.begin());
    //std::cout << "gNorm0: " << gNorm/P << std::endl;
    if(gNorm/P>eps)
    {
        LSum=sum_max_tab(LTab,m,LMax);
        if(LSum<__DBL_EPSILON__){eta=learning_rate_init;}
        else{eta=(2*(1-lamb))/LSum;}
        update(L,nbNeurons,globalIndices,weights,bias,-eta*g);
    }

    //general epoch
    while((gNorm/P>eps || dist/P>eps) and epoch<=maxEpoch)
    {   
        /* std::sort(permut.begin(), permut.end(),
        [&](const int& a, const int& b) {
            return (LTab[a] < LTab[b]);
        }
        ); */

        dist=0;
        gSum.setZero(); R_epoch=R;
        for(int i=0; i<m; i++)
        {

            taille = selection_data(permut[i],m,batch_size,P,X,Y,echantillonX,echantillonY); As[0]=echantillonX;

            if(i<m-1)
            {
                fforward(L,taille,nbNeurons,activations,weightsInter,biasInter,As,slopes);
                backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weightsInter,biasInter,As,slopes,gs,type_perte,false);
            }

            fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes);
            costPrec = risk(echantillonY,taille,As[L],type_perte,false); total_iterLoop+=1;
            R-=R_tab[permut[i]]; R+=costPrec; R_tab[permut[i]]=costPrec;

            backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad,type_perte,false);

            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());

            gSum+=grad; 
            if(i<m-1){g-=gs; g+=grad;}
            else{g=gSum;}
            grad_square=grad.squaredNorm(); gNorm=g.norm(); prod=g.dot(grad);

            eta0=eta_start; eta1=eta_start;
            condition=(grad_square>eps*eps);
            iterLoop=0;
            while(condition)
            {
                update(L,nbNeurons,globalIndices,weights,bias,-eta0*grad);
                fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
                condition=(cost-costPrec>-lamb*eta0*grad_square);
                if(condition){eta0/=f1;}
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                iterLoop+=1;
            }
            if(iterLoop>1 && eta0>__DBL_EPSILON__)
            {
                gauche = std::log10(eta0); droite = std::log10(f1*eta0);
                for (int k=0; k<nLoops; k++)
                {
                    milieu=(gauche+droite)/2; eta0=std::pow(10,milieu);
                    update(L,nbNeurons,globalIndices,weights,bias,-eta0*grad);
                    fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,taille,As[L],type_perte,false);
                    if(cost-costPrec>-lamb*eta0*grad_square)
                    {
                        m_best=gauche;
                        droite=milieu;
                        last_pass=false;
                    }
                    else
                    {
                        gauche=milieu;
                        last_pass=true;
                    }
                    iterLoop++;
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                }
                if(!last_pass){eta0=std::pow(10,m_best);}
            }
            total_iterLoop+=iterLoop;

            imax=indice_max_tab(LTab,m);

            if(prod>__DBL_EPSILON__ && permut[i]==imax && prod<gNorm*gNorm)
            {   
                condition=true;
                iterLoop=0;
                while(condition)
                {
                    update(L,nbNeurons,globalIndices,weights,bias,-eta1*g);
                    fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
                    condition=(cost-costPrec>-lamb*eta1*prod);
                    if(condition){eta1/=f1;}
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    iterLoop+=1;
                }
                if(iterLoop>1 && eta1>__DBL_EPSILON__)
                {
                    gauche = std::log10(eta1); droite = std::log10(f1*eta1);
                    for (int k=0; k<nLoops; k++)
                    {
                        milieu=(gauche+droite)/2; eta1=std::pow(10,milieu);
                        update(L,nbNeurons,globalIndices,weights,bias,-eta1*g);
                        fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,taille,As[L],type_perte,false);
                        if(cost-costPrec>-lamb*eta1*prod)
                        {
                            m_best=gauche;
                            droite=milieu;
                            last_pass=false;
                        }
                        else
                        {
                            gauche=milieu;
                            last_pass=true;
                        }
                        iterLoop++;
                        std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    }
                     if(!last_pass){eta1=std::pow(10,m_best);}
                }
                total_iterLoop+=iterLoop;
                eta=max(eta0,eta1);
            }
            else{eta=eta0;}

            if(grad_square<std::pow(eps,2) || eta<__DBL_EPSILON__){LTab[permut[i]]=0;}
            else{LTab[permut[i]]=(2*(1-lamb))/eta;}
            
            LSum=sum_max_tab(LTab,m,LMax);

            if(LSum<__DBL_EPSILON__){eta=learning_rate_init; eta_start=learning_rate_init;}
            else
            {
                eta_start=f2*(2*(1-lamb))/LMax;
                if(coeff==coeff_max){eta=2/(coeff*LSum);}
                else{eta=2/(heuris*coeff*LSum);}
            }
            
            if(grad_square>eps*eps && eta0>__DBL_EPSILON__){dist+=(2*(1-lamb)/eta0)*eta*gNorm;}

            if(i==permut[0]){LMax_now=LTab[permut[0]]; imax_now=permut[0];}
            else
            {
                if(LTab[permut[i]]>LMax_now){LMax_now=LTab[permut[i]]; imax_now=permut[i];}
            }

            if(permut[i]==imax_now){std::copy(weights.begin(),weights.end(),weightsMemory.begin()); std::copy(bias.begin(),bias.end(),biasMemory.begin());}

            update(L,nbNeurons,globalIndices,weights,bias,-eta*g);
            permut[i]=i;
        }
        std::copy(weightsMemory.begin(),weightsMemory.end(),weightsInter.begin()); std::copy(biasMemory.begin(),biasMemory.end(),biasInter.begin());

        epoch++;

        if(dist<gNorm){heuris=(heuris+heuris_max)/2;}
        else{heuris=(1+heuris)/2;}
        if(R-R_epoch<0)
        {  
            coeff/=heuris;
        }
        else
        {
            coeff=coeff_max;
        }

        /* std::cout << "gNorm: " << gNorm/P << std::endl;
        std::cout << "dist: " << dist/P << std::endl;
        std::cout << "R: " << R/P << std::endl;
        std::cout << "coeff: " << coeff << std::endl;
        std::cout << "heuris: " << heuris << std::endl; */
    }

    delete [] LTab; delete [] R_tab;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    As[0]=X; fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost=risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gNorm/P; study["finalCost"]=cost; study["time"]=double(time);
    study["total_iterLoop"]=double(total_iterLoop);

    return study;
}

std::map<std::string,double> RAG_L_ancien(Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte,
double const& learning_rate_init, int const& batch_size, double const& f1, double const& f2, double const& eps, int const& maxEpoch,
bool const tracking, bool const record, std::string const fileExtension)
{
    std::ofstream weightsFlux(("Record/weights_RAG_L_ancien"+fileExtension+".csv").c_str());
    if(!weightsFlux){std::cout << "Impossible d'ouvrir le fichier" << std::endl;}

    int N=globalIndices[2*L-1], P=X.cols(), epoch=0, iterLoop=0, total_iterLoop=0, l;
    assert(batch_size<=P);
    int m, taille;
    if(P%batch_size==0){m=P/batch_size;}
    else{m=P/batch_size+1;}
    bool condition=false;

    double eta, eta0, eta1, eta_start=learning_rate_init, cost, costPrec;
    double const lamb=0.5;
    double LSum, LMax, dist=0, grad_square=0, gNorm=1000, prod;
    double R=0, R_epoch;
    int imax, imax_now;

    double gauche, droite, milieu, m_best;
    int const nLoops=2; bool last_pass=false;

    double const coeff_max=4*m-1, heuris_max=1;
    double coeff = coeff_max, heuris=1;
    
    double *LTab = new double[m], *R_tab = new double[m];
    std::vector<int> permut(m);
    Eigen::VectorXd gSum(N), gs(N);

    Eigen::MatrixXd echantillonX, echantillonY;
    std::vector<Eigen::MatrixXd> As(L+1);
    std::vector<Eigen::MatrixXd> slopes(L);
    Eigen::VectorXd grad(N), g=Eigen::VectorXd::Zero(N);

    std::vector<Eigen::MatrixXd> weightsPrec(L);
    std::vector<Eigen::VectorXd> biasPrec(L);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //initialization
    std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());
    for(int i=0; i<m; i++)
    {   
        permut[i]=i;

        taille = selection_data(i,m,batch_size,P,X,Y,echantillonX,echantillonY); As[0]=echantillonX;
        fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes);
        backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad,type_perte,false);
        costPrec = risk(echantillonY,taille,As[L],type_perte,false); total_iterLoop+=1;
        R_tab[i]=costPrec; R+=costPrec;

        g+=grad;
        grad_square=grad.squaredNorm();

        eta=learning_rate_init;
        condition=(grad_square>eps*eps);
        while(condition)
        {
            update(L,nbNeurons,globalIndices,weights,bias,-eta*grad);
            fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
            condition=(cost-costPrec>-lamb*eta*grad_square);
            if(condition){eta/=f1;}
            std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
            total_iterLoop+=1;
        }
        
        if(grad_square<std::pow(eps,2) || eta<__DBL_EPSILON__){LTab[i]=0;}
        else{LTab[i]=(2*(1-lamb))/eta;}
    }
    gNorm=g.norm();
    //std::cout << "gNorm0: " << gNorm/P << std::endl;
    if(gNorm/P>eps)
    {
        LSum=sum_max_tab(LTab,m,LMax);
        if(LSum<__DBL_EPSILON__){eta=learning_rate_init;}
        else{eta=(2*(1-lamb))/LSum;}
        update(L,nbNeurons,globalIndices,weights,bias,-eta*g);
    }

    //general epoch
    while((gNorm/P>eps || dist/P>eps) and epoch<=maxEpoch)
    {   
        /* std::sort(permut.begin(), permut.end(),
        [&](const int& a, const int& b) {
            return (LTab[a] < LTab[b]);
        }
        ); */

        dist=0;
        gSum.setZero(); R_epoch=R;
        for(int i=0; i<m; i++)
        {

            taille = selection_data(permut[i],m,batch_size,P,X,Y,echantillonX,echantillonY); As[0]=echantillonX;

            if(i<m-1)
            {
                fforward(L,taille,nbNeurons,activations,weightsPrec,biasPrec,As,slopes);
                backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weightsPrec,biasPrec,As,slopes,gs,type_perte,false);
            }

            fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes);
            costPrec = risk(echantillonY,taille,As[L],type_perte,false); total_iterLoop+=1;
            R-=R_tab[permut[i]]; R+=costPrec; R_tab[permut[i]]=costPrec;

            backward(echantillonY,L,taille,nbNeurons,activations,globalIndices,weights,bias,As,slopes,grad,type_perte,false);

            std::copy(weights.begin(),weights.end(),weightsPrec.begin()); std::copy(bias.begin(),bias.end(),biasPrec.begin());

            gSum+=grad; 
            if(i<m-1){g-=gs; g+=grad;}
            else{g=gSum;}
            grad_square=grad.squaredNorm(); gNorm=g.norm(); prod=g.dot(grad);

            eta0=eta_start; eta1=eta_start;
            condition=(grad_square>eps*eps);
            iterLoop=0;
            while(condition)
            {
                update(L,nbNeurons,globalIndices,weights,bias,-eta0*grad);
                fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
                condition=(cost-costPrec>-lamb*eta0*grad_square);
                if(condition){eta0/=f1;}
                std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                iterLoop+=1;
            }
            if(iterLoop>1 && eta0>__DBL_EPSILON__)
            {
                gauche = std::log10(eta0); droite = std::log10(f1*eta0);
                for (int k=0; k<nLoops; k++)
                {
                    milieu=(gauche+droite)/2; eta0=std::pow(10,milieu);
                    update(L,nbNeurons,globalIndices,weights,bias,-eta0*grad);
                    fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,taille,As[L],type_perte,false);
                    if(cost-costPrec>-lamb*eta0*grad_square)
                    {
                        m_best=gauche;
                        droite=milieu;
                        last_pass=false;
                    }
                    else
                    {
                        gauche=milieu;
                        last_pass=true;
                    }
                    iterLoop++;
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                }
                if(!last_pass){eta0=std::pow(10,m_best);}
            }
            total_iterLoop+=iterLoop;

            imax=indice_max_tab(LTab,m);

            if(prod>__DBL_EPSILON__ && permut[i]==imax && prod<gNorm*gNorm)
            {   
                condition=true;
                iterLoop=0;
                while(condition)
                {
                    update(L,nbNeurons,globalIndices,weights,bias,-eta1*g);
                    fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost=risk(echantillonY,taille,As[L],type_perte,false);
                    condition=(cost-costPrec>-lamb*eta1*prod);
                    if(condition){eta1/=f1;}
                    std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    iterLoop+=1;
                }
                if(iterLoop>1 && eta1>__DBL_EPSILON__)
                {
                    gauche = std::log10(eta1); droite = std::log10(f1*eta1);
                    for (int k=0; k<nLoops; k++)
                    {
                        milieu=(gauche+droite)/2; eta1=std::pow(10,milieu);
                        update(L,nbNeurons,globalIndices,weights,bias,-eta1*g);
                        fforward(L,taille,nbNeurons,activations,weights,bias,As,slopes); cost = risk(echantillonY,taille,As[L],type_perte,false);
                        if(cost-costPrec>-lamb*eta1*prod)
                        {
                            m_best=gauche;
                            droite=milieu;
                            last_pass=false;
                        }
                        else
                        {
                            gauche=milieu;
                            last_pass=true;
                        }
                        iterLoop++;
                        std::copy(weightsPrec.begin(),weightsPrec.end(),weights.begin()); std::copy(biasPrec.begin(),biasPrec.end(),bias.begin());
                    }
                     if(!last_pass){eta1=std::pow(10,m_best);}
                }
                total_iterLoop+=iterLoop;
                eta=max(eta0,eta1);
            }
            else{eta=eta0;}

            if(grad_square<std::pow(eps,2) || eta<__DBL_EPSILON__){LTab[permut[i]]=0;}
            else{LTab[permut[i]]=(2*(1-lamb))/eta;}
            
            LSum=sum_max_tab(LTab,m,LMax);

            if(LSum<__DBL_EPSILON__){eta=learning_rate_init; eta_start=learning_rate_init;}
            else
            {
                eta_start=f2*(2*(1-lamb))/LMax;
                if(coeff==coeff_max){eta=2/(coeff*LSum);}
                else{eta=2/(heuris*coeff*LSum);}
            }
            
            if(grad_square>eps*eps && eta0>__DBL_EPSILON__){dist+=(2*(1-lamb)/eta0)*eta*gNorm;}

            update(L,nbNeurons,globalIndices,weights,bias,-eta*g);
            permut[i]=i;
        }

        epoch++;

        if(dist<gNorm){heuris=(heuris+heuris_max)/2;}
        else{heuris=(1+heuris)/2;}
        if(R-R_epoch<0)
        {  
            coeff/=heuris;
        }
        else
        {
            coeff=coeff_max;
        }

        /* std::cout << "gNorm: " << gNorm/P << std::endl;
        std::cout << "dist: " << dist/P << std::endl;
        std::cout << "R: " << R/P << std::endl;
        std::cout << "coeff: " << coeff << std::endl;
        std::cout << "heuris: " << heuris << std::endl; */
    }

    delete [] LTab; delete [] R_tab;

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    unsigned int time = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    As[0]=X; fforward(L,P,nbNeurons,activations,weights,bias,As,slopes); cost=risk(Y,P,As[L],type_perte);

    std::map<std::string,double> study;
    study["epoch"]=double(epoch); study["finalGradient"]=gNorm/P; study["finalCost"]=cost; study["time"]=double(time);
    study["total_iterLoop"]=double(total_iterLoop);

    return study;
}

std::map<std::string,double> train_Incremental(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
std::vector<std::string> const& activations, std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd>& bias, std::string const& type_perte, std::string const& algo,
double const& learning_rate_init, double const& beta1, double const& beta2, int const& batch_size, double const& mu_init, double const& seuil, double const& eps, int const& maxEpoch,
bool const tracking, bool const track_continuous, bool const record, std::string const fileExtension)
{
    std::map<std::string,double> study;

    if(algo=="RAG")
    {
        study = RAG(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,30,10000,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="RAG_L")
    {
        study = RAG_L(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,30,10000,eps,maxEpoch,tracking,record,fileExtension);
    }
    else if(algo=="RAG_L_ancien")
    {
        study = RAG_L_ancien(X,Y,L,nbNeurons,globalIndices,activations,weights,bias,type_perte,learning_rate_init,batch_size,30,10000,eps,maxEpoch,tracking,record,fileExtension);
    }

    return study;
}