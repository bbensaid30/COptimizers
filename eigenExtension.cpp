#include "eigenExtension.h"

Eigen::SMatrixXd convertToShaman(Eigen::MatrixXd const& Md)
{
    int nbRows = Md.rows(), nbCols = Md.cols();
    Eigen::SMatrixXd MS(nbRows,nbCols);

    for(int i=0; i<nbRows; i++)
    {
        for(int j=0; j<nbCols; j++)
        {
            MS(i,j) = Sdouble(Md(i,j));
        }
    }

    return MS;
}

Eigen::MatrixXd convertToDouble(Eigen::SMatrixXd const& Md)
{
    int nbRows = Md.rows(), nbCols = Md.cols();
    Eigen::MatrixXd M(nbRows,nbCols);

    for(int i=0; i<nbRows; i++)
    {
        for(int j=0; j<nbCols; j++)
        {
            M(i,j) = double(Md(i,j));
        }
    }

    return M;
}

Sdouble accumul(std::vector<Sdouble> const& values)
{
    int taille = values.size();
    Sdouble sum=0;
    for(int i=0; i<taille; i++)
    {
        sum+=values[i];
    }
    return sum;
}

Sdouble InnerProduct(std::vector<Sdouble> const& values1, std::vector<Sdouble> const& values2)
{
    int taille = values1.size();
    Sdouble sum=0;
    for(int i=0; i<taille; i++)
    {
        sum+=values1[i]*values2[i];
    }
    return sum;
}

Sdouble minimum(Sdouble const& a, Sdouble const& b)
{
    if(std::signbit((a-b).number)){return a;}
    else{return b;}
}

Sdouble maximum(Sdouble const& a, Sdouble const& b)
{
    if(std::signbit((a-b).number)){return b;}
    else{return a;}
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

double digits(double const& number, double const& error)
{
    if (error == 0)
    {
        // no error -> theorically infinite precision
        return INFINITY;
    }
    else if (std::isnan(error))
    {
        // if the error reached nan, we can trust no digit
        return 0;
    }
    else if (number == 0)
    {
        // we count the number of significant zeroes
        return std::max(0.0, -std::log10(std::abs(error)) - 1);
    }
    else
    {
        double relativeError = std::abs(error / number);

        if (relativeError >= 1)
        {
            return 0;
        }
        else
        {
            return -std::log10(relativeError);
        }
    }
}

std::string numericalNoiseDetailed(Sdouble const& n)
{
    int fdigits = std::floor(n.digits());

    if (!std::isfinite(n.number)) // not a traditional number
    {
        return "notFinite";
    }
    else if (std::isnan(n.error))
    {
        return "Nan";
    }
    else if (fdigits==0) // no significant digits
    {
        // the first zeros might be significant
        int dig = static_cast<int>(digits(0,n.error));

        if ((std::abs(n.number) >= 1) || (dig <= 0))
        {
            return "noMeaning";
        }
        else
        {
            // some zeros are significant
            return "partiallySignificant";
        }
    }
    else // a perfectly fine number
    {
        return "significant";
    }
}

bool numericalNoise(Sdouble const& n)
{
    std::string const message = numericalNoiseDetailed(n);

    if(message!="significant"){return true;}
    else{return false;}
}
