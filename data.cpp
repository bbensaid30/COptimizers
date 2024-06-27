#include "data.h"

//-------------------------------------------- Real defined functions ---------------------------------------------------------------------

std::vector<Eigen::MatrixXd> sineWave(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(1,nbPoints);
    Eigen::MatrixXd Y(1,nbPoints);
    X.row(0) = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);

    data[0] = X;
    data[1] = 0.5+0.25*(3*M_PI*X.array()).sin();

    return data;
}

std::vector<Eigen::MatrixXd> squareWave(int const& nbPoints, double const frequence)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(1,nbPoints);
    X.row(0) = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);

    data[0] = X;
    data[1] = 2*(2*(frequence*X.array()).floor()-(2*frequence*X.array()).floor())+1;

    return data;
}

std::vector<Eigen::MatrixXd> sinc1(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(1,nbPoints);
    Eigen::MatrixXd Y(1,nbPoints);
    X.row(0) = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);

    for (int i=0;i<nbPoints;i++)
    {
        if(X(0,i)<std::pow(10,-16))
        {
            Y(0,i)=1;
        }
        else
        {
            Y(0,i)=std::sin(X(0,i))/X(0,i);
        }
    }

    data[0]=X;
    data[1]=Y;

    return data;
}

std::vector<Eigen::MatrixXd> square(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(1,nbPoints);
    Eigen::MatrixXd Y(1,nbPoints);
    X.row(0) = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);

    data[0] = X;
    data[1] = X.array().pow(2);

    return data;
}

std::vector<Eigen::MatrixXd> squareRoot(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(1,nbPoints);
    Eigen::MatrixXd Y(1,nbPoints);
    X.row(0) = Eigen::ArrayXd::LinSpaced(nbPoints,0,2);

    data[0] = X;
    data[1] = X.array().sqrt();

    return data;
}

std::vector<Eigen::MatrixXd> exp(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(1,nbPoints);
    Eigen::MatrixXd Y(1,nbPoints);
    X.row(0) = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);

    data[0] = X;
    data[1] = X.array().exp();

    return data;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

std::vector<Eigen::MatrixXd> sinc2(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(2,nbPoints*nbPoints);
    Eigen::MatrixXd Y(1,nbPoints*nbPoints);
    Eigen::ArrayXd points = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);
    int i,j=0;
    double norme;

    while (j<nbPoints*nbPoints)
    {
        for (i=0;i<nbPoints;i++)
        {
            X(0,j) = points[j/nbPoints];
            X(1,j) = points[i];
            norme = std::sqrt(std::pow(X(0,j),2)+std::pow(X(1,j),2));
            if (norme < std::pow(10,-16)){Y(0,j) = 1;}
            else {Y(0,j) = std::sin(norme)/norme;}
            j++;
        }
    }

    data[0] = X;
    data[1] = Y;

    return data;
}

std::vector<Eigen::MatrixXd> exp2(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(2,nbPoints*nbPoints);
    Eigen::MatrixXd Y(1,nbPoints*nbPoints);
    Eigen::ArrayXd points = Eigen::ArrayXd::LinSpaced(nbPoints,0,4);
    int i,j=0;
    double norme;

    while (j<nbPoints*nbPoints)
    {
        for (i=0;i<nbPoints;i++)
        {
            X(0,j) = points[j/nbPoints];
            X(1,j) = points[i];
            Y(0,j) = 4*std::exp(-0.15*std::pow((X(0,j)-4),2)-0.5*std::pow(X(1,j)-3,2));
            j++;
        }
    }

    data[0] = X;
    data[1] = Y;

    return data;
}

std::vector<Eigen::MatrixXd> carreFunction2(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(2,nbPoints*nbPoints);
    Eigen::MatrixXd Y(2,nbPoints*nbPoints);
    Eigen::ArrayXd points = Eigen::ArrayXd::LinSpaced(nbPoints,-1,1);
    int i,j=0;
    double norme;

    while (j<nbPoints*nbPoints)
    {
        for (i=0;i<nbPoints;i++)
        {
            X(0,j) = points[j/nbPoints];
            X(1,j) = points[i];
            Y(0,j) = std::pow(X(0,j),2);
            Y(1,j) = std::pow(X(1,j),2);
            j++;
        }
    }

    data[0] = X;
    data[1] = Y;

    return data;
}

std::vector<Eigen::MatrixXd> twoSpiral(int const& nbPoints)
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(2,2*nbPoints);
    Eigen::MatrixXd Y(1,2*nbPoints);

    Eigen::ArrayXd theta = Eigen::ArrayXd::LinSpaced(nbPoints,0,2*M_PI);
    Eigen::ArrayXd r0 = 2*theta+M_PI;
    int i;
    Eigen::Rand::Vmt19937_64 gen{ 100 };

    for(i=0;i<nbPoints;i++)
    {
        X(0,i) = std::cos(theta[i])*r0[i];
        X(1,i) = std::sin(theta[i])*r0[i];
        Y(0,i)=0;
    }
    for(i=nbPoints;i<2*nbPoints;i++)
    {
        X(0,i) = -std::cos(theta[i-nbPoints])*r0[i-nbPoints];
        X(1,i) = -std::sin(theta[i-nbPoints])*r0[i-nbPoints];
        Y(0,i)=1;
    }

    data[0]=X; data[1]=Y;
    return data;

}

std::vector<Eigen::MatrixXd> twoSpiralOriginal()
{
    std::vector<Eigen::MatrixXd> data(2);
    Eigen::MatrixXd X(2,2*97);
    Eigen::MatrixXd Y(1,2*97);

    Eigen::ArrayXd phi = Eigen::ArrayXd::LinSpaced(97,0,96);
    Eigen::ArrayXd r = 6.5*(104-phi)/104;
    phi*=M_PI/16;
    int i;
    //Eigen::Rand::Vmt19937_64 gen{ 100 };

    for(i=0;i<97;i++)
    {
        X(0,i) = std::cos(phi[i])*r[i];
        X(1,i) = std::sin(phi[i])*r[i];
        Y(0,i)=0;
    }
    for(i=97;i<2*97;i++)
    {
        X(0,i) = -std::cos(phi[i-97])*r[i-97];
        X(1,i) = -std::sin(phi[i-97])*r[i-97];
        Y(0,i)=1;
    }

    data[0]=X; data[1]=Y;
    return data;

}

std::vector<Eigen::MatrixXd> Boston(int const PTrain, int const PTest)
{
    std::ifstream fileTrainInputs("Data/boston_inputs_train.csv");
    std::ifstream fileTrainOutputs("Data/boston_outputs_train.csv");
    std::ifstream fileTestInputs("Data/boston_inputs_test.csv");
    std::ifstream fileTestOutputs("Data/boston_outputs_test.csv");

    Eigen::MatrixXd XTrain(13,PTrain), XTest(13,PTest);
    Eigen::MatrixXd YTrain = Eigen::MatrixXd::Zero(1,PTrain), YTest = Eigen::MatrixXd::Zero(1,PTest);
    std::vector<Eigen::MatrixXd> data(4);

    std::string line, subChaine;
    std::stringstream ss(line);
    int i=0,j=0;

   if(fileTrainInputs && fileTrainOutputs)
   {

      while(getline(fileTrainInputs, line) && i<PTrain)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTrain(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTrainOutputs, line) && i<PTrain)
      {
            YTrain(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier d'entraînement en lecture." << std::endl;
   }

   i=0;j=0;

   if(fileTestInputs && fileTestOutputs)
   {

      while(getline(fileTestInputs, line) && i<PTest)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTest(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTestOutputs, line) && i<PTest)
      {
            YTest(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier de test en lecture." << std::endl;
   }

   data[0]=XTrain; data[1]=YTrain; data[2]=XTest; data[3]=YTest;

   return data;

}

std::vector<Eigen::MatrixXd> Sonar(int const PTrain, int const PTest)
{
    std::ifstream fileTrainInputs("Data/sonar_inputs_train.csv");
    std::ifstream fileTrainOutputs("Data/sonar_outputs_train.csv");
    std::ifstream fileTestInputs("Data/sonar_inputs_test.csv");
    std::ifstream fileTestOutputs("Data/sonar_outputs_test.csv");

    Eigen::MatrixXd XTrain(60,PTrain), XTest(60,PTest);
    Eigen::MatrixXd YTrain = Eigen::MatrixXd::Zero(1,PTrain), YTest = Eigen::MatrixXd::Zero(1,PTest);
    std::vector<Eigen::MatrixXd> data(4);

    std::string line, subChaine;
    std::stringstream ss(line);
    int i=0,j=0;

   if(fileTrainInputs && fileTrainOutputs)
   {

      while(getline(fileTrainInputs, line) && i<PTrain)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTrain(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTrainOutputs, line) && i<PTrain)
      {
            YTrain(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier d'entraînement en lecture." << std::endl;
   }

   i=0;j=0;

   if(fileTestInputs && fileTestOutputs)
   {

      while(getline(fileTestInputs, line) && i<PTest)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTest(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTestOutputs, line) && i<PTest)
      {
            YTest(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier de test en lecture." << std::endl;
   }

   data[0]=XTrain; data[1]=YTrain; data[2]=XTest; data[3]=YTest;

   return data;

}

std::vector<Eigen::MatrixXd> German(int const PTrain, int const PTest)
{
    std::ifstream fileTrainInputs("Data/german_inputs_train.csv");
    std::ifstream fileTrainOutputs("Data/german_outputs_train.csv");
    std::ifstream fileTestInputs("Data/german_inputs_test.csv");
    std::ifstream fileTestOutputs("Data/german_outputs_test.csv");

    Eigen::MatrixXd XTrain(26,PTrain), XTest(26,PTest);
    Eigen::MatrixXd YTrain = Eigen::MatrixXd::Zero(1,PTrain), YTest = Eigen::MatrixXd::Zero(1,PTest);
    std::vector<Eigen::MatrixXd> data(4);

    std::string line, subChaine;
    std::stringstream ss(line);
    int i=0,j=0;

   if(fileTrainInputs && fileTrainOutputs)
   {

      while(getline(fileTrainInputs, line) && i<PTrain)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTrain(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTrainOutputs, line) && i<PTrain)
      {
            YTrain(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier d'entraînement en lecture." << std::endl;
   }

   i=0;j=0;

   if(fileTestInputs && fileTestOutputs)
   {

      while(getline(fileTestInputs, line) && i<PTest)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTest(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTestOutputs, line) && i<PTest)
      {
            YTest(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier de test en lecture." << std::endl;
   }

   data[0]=XTrain; data[1]=YTrain; data[2]=XTest; data[3]=YTest;

   return data;

}

std::vector<Eigen::MatrixXd> Diamonds(int const PTrain, int const PTest)
{
    std::ifstream fileTrainInputs("Data/diamonds_inputs_train.csv");
    std::ifstream fileTrainOutputs("Data/diamonds_outputs_train.csv");
    std::ifstream fileTestInputs("Data/diamonds_inputs_test.csv");
    std::ifstream fileTestOutputs("Data/diamonds_outputs_test.csv");

    Eigen::MatrixXd XTrain(9,PTrain), XTest(9,PTest);
    Eigen::MatrixXd YTrain = Eigen::MatrixXd::Zero(1,PTrain), YTest = Eigen::MatrixXd::Zero(1,PTest);
    std::vector<Eigen::MatrixXd> data(4);

    std::string line, subChaine;
    std::stringstream ss(line);
    int i=0,j=0;

   if(fileTrainInputs && fileTrainOutputs)
   {

      while(getline(fileTrainInputs, line) && i<PTrain)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTrain(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTrainOutputs, line) && i<PTrain)
      {
            YTrain(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier d'entraînement en lecture." << std::endl;
   }

   i=0;j=0;

   if(fileTestInputs && fileTestOutputs)
   {

      while(getline(fileTestInputs, line) && i<PTest)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTest(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTestOutputs, line) && i<PTest)
      {
            YTest(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier de test en lecture." << std::endl;
   }

   data[0]=XTrain; data[1]=YTrain; data[2]=XTest; data[3]=YTest;

   return data;

}

std::vector<Eigen::MatrixXd> California(int const PTrain, int const PTest)
{
    std::ifstream fileTrainInputs("Data/california_inputs_train.csv");
    std::ifstream fileTrainOutputs("Data/california_outputs_train.csv");
    std::ifstream fileTestInputs("Data/california_inputs_test.csv");
    std::ifstream fileTestOutputs("Data/california_outputs_test.csv");

    Eigen::MatrixXd XTrain(16,PTrain), XTest(16,PTest);
    Eigen::MatrixXd YTrain = Eigen::MatrixXd::Zero(1,PTrain), YTest = Eigen::MatrixXd::Zero(1,PTest);
    std::vector<Eigen::MatrixXd> data(4);

    std::string line, subChaine;
    std::stringstream ss(line);
    int i=0,j=0;

   if(fileTrainInputs && fileTrainOutputs)
   {

      while(getline(fileTrainInputs, line) && i<PTrain)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTrain(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTrainOutputs, line) && i<PTrain)
      {
            YTrain(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier d'entraînement en lecture." << std::endl;
   }

   i=0;j=0;

   if(fileTestInputs && fileTestOutputs)
   {

      while(getline(fileTestInputs, line) && i<PTest)
      {
            ss=(std::stringstream)line;
            while (std::getline(ss, subChaine, ','))
            {
                XTest(j,i)=double(strtod(subChaine.c_str(),NULL));
                j++;
            }
            i++; j=0;
      }

      i=0;
      while(getline(fileTestOutputs, line) && i<PTest)
      {
            YTest(0,i)=double(strtod(line.c_str(),NULL));
            i++;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier de test en lecture." << std::endl;
   }

   data[0]=XTrain; data[1]=YTrain; data[2]=XTest; data[3]=YTest;

   return data;

}

std::vector<Eigen::MatrixXd> MNIST(int const PTrain, int const PTest)
{
    std::ifstream fileTrain("Data/mnist_train.csv");
    std::ifstream fileTest("Data/mnist_test.csv");

    Eigen::MatrixXd XTrain(784,PTrain), XTest(784,PTest);
    Eigen::MatrixXd YTrain = Eigen::MatrixXd::Zero(10,PTrain), YTest = Eigen::MatrixXd::Zero(10,PTest);
    std::vector<Eigen::MatrixXd> data(4);

    std::string line, subChaine;
    std::stringstream ss(line);
    int i=0,j=0;

   if(fileTrain)
   {

      while(getline(fileTrain, line) && i<PTrain)
      {
            ss=(std::stringstream)line;
            std::getline(ss, subChaine, ','); YTrain(strtol(subChaine.c_str(),NULL,10), i)=1;
            while (std::getline(ss, subChaine, ','))
            {
                XTrain(j,i)=double(strtol(subChaine.c_str(),NULL,10))/255;
                j++;
            }
            i++; j=0;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier d'entraînement en lecture." << std::endl;
   }

   i=0;j=0;

   if(fileTest)
   {

      while(getline(fileTest, line) && i<PTest)
      {
            ss=(std::stringstream)line;
            getline(ss, subChaine, ','); YTest(strtol(subChaine.c_str(),NULL,10),i)=1;
            while (std::getline(ss, subChaine, ','))
            {
                XTest(j,i)=double(strtol(subChaine.c_str(),NULL,10))/255;
                j++;
            }
            i++; j=0;
      }
   }
   else
   {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier de test en lecture." << std::endl;
   }

   data[0]=XTrain; data[1]=YTrain; data[2]=XTest; data[3]=YTest;

   return data;

}

std::vector<Eigen::MatrixXd> trainTestData(std::vector <Eigen::MatrixXd> const& data, double const& percTrain, bool const reproductible)
{
    std::vector<Eigen::MatrixXd> dataTrainTest(4);
    int const P = data[0].cols(), n0 = data[0].rows(), nL=data[1].rows();
    int const sizeTrain = (int) (percTrain*P);
    int const sizeTest = P-sizeTrain;
    std::vector<char> nbChosen(P);
    Eigen::MatrixXd XTrain(n0,sizeTrain), XTest(n0,sizeTest), YTrain(nL,sizeTrain), YTest(nL,sizeTest);
    int i, j=0, iter=0, randomNumber;
    std::default_random_engine re(0);
    std::default_random_engine reAlea(time(0));
    std::uniform_int_distribution<int> distrib{0,P-1};

    for(i=0;i<P;i++) {nbChosen[i]='n';}

    if (reproductible){randomNumber=distrib(re);}
    else {randomNumber=distrib(reAlea);}
    nbChosen[randomNumber]='y';
    XTrain.col(0) = data[0].col(randomNumber);
    YTrain.col(0) = data[1].col(randomNumber);
    for (i=1;i<sizeTrain;i++)
    {
       iter=0;
       do
       {
        if (reproductible){randomNumber=distrib(re);}
        else {randomNumber=distrib(reAlea);}
        iter++;
       }while(nbChosen[randomNumber]=='y');
       nbChosen[randomNumber]='y';

       XTrain.col(i) = data[0].col(randomNumber);
       YTrain.col(i) = data[1].col(randomNumber);
    }

    for (i=0;i<P;i++)
    {
        if (nbChosen[i]=='n')
        {
            XTest.col(j) = data[0].col(i);
            YTest.col(j) = data[1].col(i);
            j++;
        }
    }

    dataTrainTest[0] = XTrain; dataTrainTest[1] = YTrain; dataTrainTest[2] = XTest; dataTrainTest[3] = YTest;

    return dataTrainTest;
}
