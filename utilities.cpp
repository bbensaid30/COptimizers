#include "utilities.h"

int cyclic(int i, int m)
{
    if(i<m-1){return i+1;}
    else{return 0;}
}

int indice_max_tab(double *tab, int const& taille)
{   
    double max=tab[0]; int imax=0;
    for(int i=1; i<taille; i++)
    {
        if(tab[i]>max)
        {
            max=tab[i];
            imax=i;
        }
    }
    return imax;
}

double sum_max_tab(double *tab, int const& taille, double& max)
{   
    double sum=tab[0]; max=tab[0];
    for(int i=1; i<taille; i++)
    {
        if(tab[i]>max){max=tab[i];}
        sum+=tab[i];
    }
    return sum;
}

double sum_tab(double *tab, int const& taille)
{   
    double sum=tab[0];
    for(int i=1; i<taille; i++)
    {
        sum+=tab[i];
    }
    return sum;
}

double sum_decale(double *tab, int const& taille, int i)
{   
    double sum=0;
    for(int j=0; j<taille; j++)
    {
        sum+=(j-i+taille)*tab[j];
    }
    return sum;
}

double mean_tab(double *tab, int const& taille)
{   
    double avg=0;
    for(int i=0; i<taille; i++)
    {
        avg+=tab[i];
    }
    return avg/taille;
}

void affiche_tab(double *tab, int const& taille)
{
    for(int i=0; i<taille; i++)
    {
        std::cout << "tab[" << i << "] " << tab[i] << std::endl;
    }
    std::cout << "---------------------" << std::endl;
}

void init_grad_zero(std::vector<Eigen::VectorXd>& grads, int m, int N)
{
    for (int i=0; i<m; i++)
    {
        grads[i] = Eigen::VectorXd::Zero(N);
    }
}

int selection_data(int const& i, int const& m, int const& batch_size, int const& P, Eigen::MatrixXd const& X, Eigen::MatrixXd const& Y, Eigen::MatrixXd& echantillonX, Eigen::MatrixXd& echantillonY)
{
    int numero=i*batch_size;
    if(i<m-1)
    {
        echantillonX = X.middleCols(numero,batch_size);
        echantillonY = Y.middleCols(numero,batch_size);
        return batch_size;
    }
    else
    {
        echantillonX = X.middleCols(numero, P-numero);
        echantillonY = Y.middleCols(numero,P-numero);
        return P-numero;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------
double max(double a, double b)
{
    if(a>=b){return a;}
    else{return b;}
}

double dInf(Eigen::MatrixXd const& A, Eigen::MatrixXd const& B)
{   
    int const P = A.cols();
    int Nthrds;
    #pragma omp parallel
    {
        #pragma omp single
        Nthrds = omp_get_num_threads();
    }

    Eigen::VectorXd normes(Nthrds);

    #pragma omp parallel
    {
        int id, istart, iend;
        id = omp_get_thread_num();
        istart = id * P / Nthrds;
        iend = (id+1) * P / Nthrds;
        if(id == Nthrds-1){iend = P;}

        normes(id) = (A.middleCols(istart,iend-istart)-B.middleCols(istart,iend-istart)).lpNorm<Eigen::Infinity>();
    }

    return normes.lpNorm<Eigen::Infinity>();
}

double MAE(Eigen::MatrixXd const& A, Eigen::MatrixXd const& B)
{   
    int const P = A.cols();
    int Nthrds;
    #pragma omp parallel
    {
        #pragma omp single
        Nthrds = omp_get_num_threads();
    }

    Eigen::VectorXd normes(Nthrds);

    #pragma omp parallel
    {
        int id, istart, iend;
        id = omp_get_thread_num();
        istart = id * P / Nthrds;
        iend = (id+1) * P / Nthrds;
        if(id == Nthrds-1){iend = P;}

        normes(id) = (A.middleCols(istart,iend-istart)-B.middleCols(istart,iend-istart)).lpNorm<1>();
    }

    return normes.lpNorm<1>()/(double)(P*A.rows());
}

double MAPE(Eigen::MatrixXd const& A, Eigen::MatrixXd const& B)
{   
    int const P = A.cols();
    int Nthrds;
    #pragma omp parallel
    {
        #pragma omp single
        Nthrds = omp_get_num_threads();
    }

    Eigen::VectorXd normes(Nthrds);

    #pragma omp parallel
    {
        int id;
        id = omp_get_thread_num();
        normes(id)=0;
        #pragma omp for
        for(int p=0; p<P; p++)
        {
            for(int i=0; i<A.rows(); i++)
            {
                if(A(i,p)<std::pow(10,-10)){std::cout << "valeur nulle" << std::endl;}
                else{normes(id) += std::abs((A(i,p)-B(i,p))/A(i,p));}
            }
        }
    }

    return normes.sum()/(P*A.rows());
}

void echanger(double& a, double& b)
{
    double inter = b;
    b=a; a=inter;
}

bool appartient_intervalle(double x, double gauche, double droite)
{
    if(gauche <= droite)
    {
        if(x>=gauche && x <=droite){return true;}
        else{return false;}
    }
    else
    {
        if(x>=droite && x <=gauche){return true;}
        else{return false;}
    }
}

int proportion(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, std::vector<double>& proportions, std::vector<double>& distances, double const& epsNeight)
{
    int const nbPoints=points.size(), nbProportions=proportions.size();
    assert(nbPoints==nbProportions);

    int i=0;
    double distance;
    for(i=0;i<nbPoints;i++)
    {
        distance=(currentPoint-points[i]).norm();
        if (distance<epsNeight)
        {
            proportions[i]++;
            distances[i]+=distance;
            return i;
        }
    }
    return -1;
}

int numero_point(Eigen::VectorXd const& currentPoint, std::vector<Eigen::VectorXd> const& points, double const& epsNeight)
{
    int const nbPoints=points.size();

    int i=0;
    double distance;
    for(i=0;i<nbPoints;i++)
    {
        distance=(currentPoint-points[i]).norm();
        if (distance<epsNeight)
        {
            return i;
        }
    }
    return -1;
}

double mean(std::vector<int> const& values)
{
    int sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();

    return mean;
}

double sd(std::vector<int> const& values, double const& moy)
{
    int sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / values.size() - std::pow(moy,2));

    return stdev;
}

double median(std::vector<double>& values)
{
        int const taille = values.size();

        if (taille == 0)
                throw std::domain_error("median of an empty vector");

        std::sort(values.begin(), values.end());

        int const mid = taille/2;

        return taille % 2 == 0 ? (values[mid] + values[mid-1]) / 2.0 : values[mid];
}

int median(std::vector<int>& values)
{
        int const taille = values.size();

        if (taille == 0)
                throw std::domain_error("median of an empty vector");

        std::sort(values.begin(), values.end());

        int const mid = taille/2;

        return taille % 2 == 0 ? (values[mid] + values[mid-1]) / 2.0 : values[mid];
}

double minVector(std::vector<double> const& values)
{
    int taille = values.size();
    if (taille==0){throw std::domain_error("minimum of an empty vector");}

    double minimum = values[0];
    for (int i=0; i<taille ; i++)
    {
        if(values[i]<minimum){minimum=values[i];}
    }

    return minimum;
}
int minVector(std::vector<int> const& values)
{
    int taille = values.size();
    if (taille==0){throw std::domain_error("minimum of an empty vector");}

    int minimum = values[0];
    for (int i=0; i<taille ; i++)
    {
        if(values[i]<minimum){minimum=values[i];}
    }

    return minimum;
}

double norme(std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::string const norm)
{
    double dis=0.0, disCurrent;
    int const L = weights.size();
    int l;

    if(norm=="infinity")
    {
        for(l=0;l<L;l++)
        {
            disCurrent = weights[l].lpNorm<Eigen::Infinity>();
            if(disCurrent>dis){dis=disCurrent;}
            disCurrent = bias[l].lpNorm<Eigen::Infinity>();
            if(disCurrent>dis){dis=disCurrent;}
        }
        return dis;
    }
    else
    {
        for(l=0;l<L;l++)
        {
            dis += weights[l].squaredNorm();
            dis += bias[l].squaredNorm();
        }
        return std::sqrt(dis);
    }

}


double distance(std::vector<Eigen::MatrixXd> const& weightsPrec, std::vector<Eigen::VectorXd> const& biasPrec,
std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, std::string const norm)
{
    double dis=0.0, disCurrent;
    int const L = weights.size();
    int l;

    if(norm=="infinity")
    {
        for(l=0;l<L;l++)
        {
            disCurrent = (weights[l]-weightsPrec[l]).lpNorm<Eigen::Infinity>();
            if(disCurrent>dis){dis=disCurrent;}
            disCurrent = (bias[l]-biasPrec[l]).lpNorm<Eigen::Infinity>();
            if(disCurrent>dis){dis=disCurrent;}
        }
        return dis;
    }
    else
    {
        for(l=0;l<L;l++)
        {
            dis += (weights[l]-weightsPrec[l]).squaredNorm();
            dis += (bias[l]-biasPrec[l]).squaredNorm();
        }
        return std::sqrt(dis);
    }

}

double cosVector(Eigen::VectorXd const& v1, Eigen::VectorXd const& v2)
{
    double result;
    result = v1.dot(v2);
    return result/=v1.norm()*v2.norm();
}

void convexCombination(std::vector<Eigen::MatrixXd>& weightsMoy, std::vector<Eigen::VectorXd>& biasMoy, std::vector<Eigen::MatrixXd> const& weights,
std::vector<Eigen::VectorXd> const& bias, int const& L, double const& lambda)
{
    for(int l=0;l<L;l++)
    {
        weightsMoy[l]=lambda*weightsMoy[l]+(1-lambda)*weights[l];
        biasMoy[l]=lambda*biasMoy[l]+(1-lambda)*bias[l];
    }
}

void RKCombination(std::vector<Eigen::MatrixXd>& weightsInter, std::vector<Eigen::VectorXd>& biasInter, std::vector<Eigen::MatrixXd> const& weights, std::vector<Eigen::VectorXd> const& bias, int const& L)
{
    for(int l=0;l<L;l++)
    {
        weightsInter[l]=2*weightsInter[l]-weights[l];
        biasInter[l]=2*biasInter[l]-bias[l];
    }
}

void nesterovCombination(std::vector<Eigen::MatrixXd> const& weights1, std::vector<Eigen::VectorXd> const& bias1, std::vector<Eigen::MatrixXd> const& weights2,
std::vector<Eigen::VectorXd> const& bias2, std::vector<Eigen::MatrixXd>& weightsInter, std::vector<Eigen::VectorXd>& biasInter, int const& L, double const& lambda)
{
    for(int l=0;l<L;l++)
    {
        weightsInter[l]= weights1[l]+lambda*(weights1[l]-weights2[l]);
        biasInter[l] = bias1[l]+lambda*(bias1[l]-bias2[l]);
    }
}

void tabToVector(std::vector<Eigen::MatrixXd>& weights, std::vector<Eigen::VectorXd> const& bias, int const& L, std::vector<int> const& nbNeurons, std::vector<int> const& globalIndices,
Eigen::VectorXd& point)
{
    int l, jump;

    for (l=0;l<L;l++)
    {
        jump=nbNeurons[l]*nbNeurons[l+1];
        weights[l].resize(jump,1);
        point.segment(globalIndices[2*l]-jump,jump)=weights[l];
        jump=nbNeurons[l+1];
        point.segment(globalIndices[2*l+1]-jump,jump)=bias[l];

        weights[l].resize(nbNeurons[l+1],nbNeurons[l]);
    }
}

void standardization(Eigen::MatrixXd& X)
{
    int const dim=X.rows(), P=X.cols();
    Eigen::VectorXd mean(dim), standardDeviation(dim);

    mean = X.rowwise().mean();
    for(int i=0; i<dim;i++)
    {
        X.array().row(i) -= mean(i);
    }

    standardDeviation = X.rowwise().squaredNorm()/(double(P));
    standardDeviation.cwiseSqrt();

    for(int i=0; i<dim;i++)
    {
        X.row(i) /= standardDeviation(i);
    }
}

int nbLines(std::ifstream& flux) {
    std::string s;

    unsigned int nb = 0;
    while(std::getline(flux,s)) {++nb;}

    return nb;

}

void readMatrix(std::ifstream& flux, Eigen::MatrixXd& result, int const& nbRows, int const& nbCols)
{
    int cols, rows;
    std::string line;

    for(rows=0; rows<nbRows; rows++)
    {
        std::getline(flux, line);

        std::stringstream stream(line);
        cols=0;
        while(! stream.eof())
        {
            stream >> result(rows,cols);
            cols++;
        }
    }

}

void readVector(std::ifstream& flux, Eigen::VectorXd& result, int const& nbRows)
{
    std::string line;

    for(int i=0; i<nbRows; i++)
    {
        std::getline(flux, line);

        std::stringstream stream(line);
        stream >> result(i);
    }

}

double indexProperValues(Eigen::MatrixXd const& H)
{
    double prop=0;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(H);
    Eigen::VectorXd eivals = eigensolver.eigenvalues();

    int const taille = eivals.rows();
    for(int i=0; i<taille; i++)
    {
        if(eivals(i)<0){prop++;}
    }
    return prop/taille;
}


double expInv(double const& x)
{
    double const eps = std::pow(10,-14);
    if(std::abs(x)<eps)
    {
        return 0;
    }
    else
    {
        return std::exp(-1/(x*x));
    }
}

double fAdam(double const& a, double const& b, double const& t)
{
    return std::sqrt(1-std::exp(-b*t))/(1-std::exp(-a*t));
}
