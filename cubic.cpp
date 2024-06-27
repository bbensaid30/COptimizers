#include "cubic.h"


//Résolution quand delta est non nul autrement dit il existe trois solutions distinctes
void ResolER3(Sdouble const& a, Sdouble const& b, Sdouble const& c, Sdouble const& d, Cmplx x[3])
{
    Sdouble delta0 = Sstd::pow(b,2)-3*a*c;
    Sdouble delta1 = 2*Sstd::pow(b,3)-9*a*b*c+27*Sstd::pow(a,2)*d;

    Cmplx C[3];
    cbrt(0.5*(Cmplx(delta1,0)+sqrt(Cmplx(Sstd::pow(delta1,2)-4*Sstd::pow(delta0,3),0))),C);

    for(int i=0; i<3;i++)
    {
        x[i] = -(b+C[i]+Cmplx(delta0,0)/C[i])/(3*a);
    }
}

Sdouble solPositive(Sdouble const& a, Sdouble const& b, Sdouble const& c, Sdouble const& d)
{
    Sdouble delta = 18*a*b*c*d-4*Sstd::pow(b,3)*d+Sstd::pow(b,2)*Sstd::pow(c,2)-4*a*Sstd::pow(c,3)-27*Sstd::pow(a,2)*Sstd::pow(d,2);
    Sdouble delta0 = Sstd::pow(b,2)-3*a*c;
    Sdouble maxi;
    Cmplx x[3];


    if(Sstd::abs(delta)+std::abs(delta.error)<std::pow(10,-16))
    {
        if(Sstd::abs(delta0)+std::abs(delta0.error)<std::pow(10,-16))
        {
//            std::cout << "delta0: " << delta0 << std::endl;
//            Sdouble test  = -b/(3*a);
//            std::cout << "test: " << a*Sstd::pow(test,3)+b*Sstd::pow(test,2)+c*test+d << std::endl;
            return -b/(3*a);
        }
        else
        {
//            std::cout << "delta0: " << delta0 << std::endl;
//            Sdouble test  = (4*a*b*c-9*Sstd::pow(a,2)*d-Sstd::pow(b,3))/(a*delta0);
//            std::cout << "test: " << a*Sstd::pow(test,3)+b*Sstd::pow(test,2)+c*test+d << std::endl;
            return (4*a*b*c-9*Sstd::pow(a,2)*d-Sstd::pow(b,3))/(a*delta0);
        }
    }
    else if(delta<0)
    {
        ResolER3(a,b,c,d,x);
        for(int i=0; i<3;i++)
        {
            if(Sstd::abs(imag(x[i]))<std::pow(10,-16)){ return real(x[i]);}
        }
    }
    else
    {
        ResolER3(a,b,c,d,x);
        maxi=real(x[0]);
        for(int i=1; i<3; i++)
        {
            if(real(x[i])>maxi){maxi = real(x[i]);}
        }
        return maxi;
    }
}



Sdouble pasAdaptatifMomentum(Sdouble const& beta_bar, Sdouble const& deltaR, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar)
{
    Sdouble a = -4*Sstd::pow(beta_bar,4)*deltaR*grad2R;
    Sdouble b = 4*Sstd::pow(beta_bar,2)*(Sstd::pow(prodScalar,2)-2*beta_bar*deltaR*grad2R+4*beta_bar*deltaR*prodScalar);
    Sdouble c = 16*beta_bar*(-norm2v*prodScalar+2*beta_bar*deltaR*prodScalar-beta_bar*deltaR*norm2v);
    Sdouble d = 16*norm2v*(norm2v-2*beta_bar*deltaR);

    if(a>0){std::cout << "ça fait mal" << std::endl; std::cout << "a: " << a << std::endl; }
    if(d<0){std::cout << "ça fait mal" << std::endl; std::cout << "d: " << d << "+- " << std::abs(d.error) << std::endl;}

    if(Sstd::abs(a)+std::abs(a.error)<std::pow(10,-16)){std::cout << "pas de degré 3 !!!!! " << std::endl;}

    return solPositive(a,b,c,d);
}

Sdouble deltah(Sdouble const& beta_bar, Sdouble const& deltaR, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar, Sdouble const& h)
{
    Sdouble a = -4*Sstd::pow(beta_bar,4)*deltaR*grad2R;
    Sdouble b = 4*Sstd::pow(beta_bar,2)*(Sstd::pow(prodScalar,2)-2*beta_bar*deltaR*grad2R+4*beta_bar*deltaR*prodScalar);
    Sdouble c = 16*beta_bar*(-norm2v*prodScalar+2*beta_bar*deltaR*prodScalar-beta_bar*deltaR*norm2v);
    Sdouble d = 16*norm2v*(norm2v-2*beta_bar*deltaR);

    Sdouble result = b+a*h; result = c+result*h; result = d+result*h;
    return result;
}

Sdouble solMomentum(Sdouble const& beta_bar, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar, Sdouble const& h, Sdouble const& delta)
{
    Sdouble a = (2+beta_bar*h)*(4*norm2v-4*beta_bar*h*prodScalar+Sstd::pow(beta_bar*h,2)*grad2R);
    Sdouble b = 2*beta_bar*h*prodScalar-4*norm2v;
    Sdouble x1,x2;

    if(Sstd::abs(delta)+std::abs(delta.error)<std::pow(10,-16))
    {
       return -b/(2*a);
    }
    else if(!std::signbit(delta.number))
    {
        x1 = (-b-Sstd::sqrt(delta))/(2*a); x2 =  (-b+Sstd::sqrt(delta))/(2*a);
        if(Sstd::abs(x1-0.5)<Sstd::abs(x2-0.5)){return x1;}
        else{return x2;}
    }
    else
    {
        std::cout << "attention delta<0: " << delta << std::endl;
        return -b/(2*a);
    }
}

Sdouble* solsMomentum(Sdouble const& beta_bar, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar, Sdouble const& h, Sdouble const& delta)
{
    Sdouble a = (2+beta_bar*h)*(4*norm2v-4*beta_bar*h*prodScalar+Sstd::pow(beta_bar*h,2)*grad2R);
    Sdouble b = 2*beta_bar*h*prodScalar-4*norm2v;
    static Sdouble x[2];

    if(std::abs(delta.number)+std::abs(delta.error)<std::pow(10,-16))
    {
       x[0]=-b/(2*a); x[1]=-b/(2*a);
    }
    else if(!std::signbit(delta.number))
    {
        x[0] = (-b-Sstd::sqrt(delta))/(2*a); x[1] =  (-b+Sstd::sqrt(delta))/(2*a);
    }
    else
    {
        std::cout << "attention delta<0: " << delta << std::endl;
    }
    return x;
}


