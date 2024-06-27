#include <math.h>
#include "Cmplx.h"

// constructeurs
Cmplx::Cmplx() {R = 0.0; I = 0.0;} // default constructor
Cmplx::Cmplx(Reel re, Reel im) {R = re; I = im;} // constructor
Cmplx::Cmplx(const Cmplx& z) {R = z.R; I = z.I;} // constructor

// affectation
Cmplx& Cmplx::operator = (const Reel r) {this->R=r; this->I=0.0; return *this;}

// affectations avec opération (Cmplx)
Cmplx& Cmplx::operator += (const Cmplx& z) {R += z.R; I += z.I; return *this;}
Cmplx& Cmplx::operator -= (const Cmplx& z) {R -= z.R; I -= z.I; return *this;}
Cmplx& Cmplx::operator *= (const Cmplx& z) {Reel r=R; R=r*z.R-I*z.I; I=r*z.I+I*z.R; return *this;}
Cmplx& Cmplx::operator /= (const Cmplx& z) {Reel r=R,a=z.R,b=z.I,m=a*a+b*b; R=(r*a+I*b)/m; I=(I*a-r*b)/m; return *this;}

// affectations avec opération (Reel)
Cmplx& Cmplx::operator += (const Reel d) {R += d; return *this;}
Cmplx& Cmplx::operator -= (const Reel d) {R -= d; return *this;}
Cmplx& Cmplx::operator *= (const Reel d) {R *= d; I *= d; return *this;}
Cmplx& Cmplx::operator /= (const Reel d) {R /= d; I /= d; return *this;}

// fonctions réelles standards
Reel real(const Cmplx& z) {return z.R;}
Reel imag(const Cmplx& z) {return z.I;}
Reel norm(const Cmplx& z) {return z.R*z.R+z.I*z.I;}
Reel abs (const Cmplx& z) {return Sstd::sqrt(norm(z));}
Reel arg (const Cmplx& z) {return Sstd::atan2(z.I,z.R);}

std::ostream& operator<<(std::ostream& os, const Cmplx& z)
{
    char *ss = "+i";
    Reel im = imag(z);
    if (im < 0.0) {ss = "-i"; im=-im;}
    os << '[' << real(z) << ss << char(250) << im <<  "]";
    return os;
}

// -------------- opérateurs de comparaison
bool operator==(const Cmplx y,const Cmplx& z) {return y.R==z.R && y.I==z.I;}
bool operator!=(const Cmplx y,const Cmplx& z) {return y.R!=z.R || y.I!=z.I;}
bool operator==(const Cmplx y,const Reel d) {return y.R==d && y.I==0;}
bool operator!=(const Cmplx y,const Reel d) {return y.R!=d || y.I!=0;}


// -------------- opérateurs unaires
Cmplx operator+(const Cmplx& z) {return z;}
Cmplx operator-(const Cmplx& z) {return Cmplx(-z.R,-z.I);}

// -------------- opérateurs binaires
Cmplx& operator+(Cmplx y,const Cmplx& z) {return y+=z;}
Cmplx& operator+(const Reel a,Cmplx z) {return z+=a;}
Cmplx& operator+(Cmplx z,const Reel a) {return z+=a;}
Cmplx& operator-(Cmplx y,const Cmplx& z) {return y-=z;}
Cmplx  operator-(const Reel a,const Cmplx& z) {return Cmplx(a-z.R,-z.I);}
Cmplx& operator-(Cmplx z,const Reel a) {return z-=a;}
Cmplx& operator*(Cmplx y,const Cmplx& z) {return y*=z;}
Cmplx& operator*(const Reel a,Cmplx z) {return z*=a;}
Cmplx& operator*(Cmplx z,const Reel a) {return z*=a;}
Cmplx& operator/(Cmplx y,const Cmplx& z) {return y/=z;}
Cmplx  operator/(Reel a,const Cmplx& z) {a/=norm(z); return Cmplx(a*z.R,-a*z.I);}
Cmplx& operator/(Cmplx z,const Reel a) {return z/=a;}

// -------------- fonctions complexes  standards
Cmplx conj(const Cmplx& z) {return Cmplx(z.R,-z.I);}
Cmplx cos (const Cmplx& z) {return Cmplx(Sstd::cos(z.R)*Sstd::cosh(z.I),-Sstd::sin(z.R)*Sstd::sinh(z.I));}
Cmplx cosh(const Cmplx& z) {return Cmplx(Sstd::cosh(z.R)*Sstd::cos(z.I),Sstd::sinh(z.R)*Sstd::sin(z.I));}
Cmplx exp (const Cmplx& z) {Reel e=Sstd::exp(z.R); return Cmplx(e*Sstd::cos(z.I),e*Sstd::sin(z.I));}
Cmplx log (const Cmplx& z) {Reel m=abs(z); return Cmplx(Sstd::log(m),2*Sstd::atan2(z.I,z.R+m));}
Cmplx log10(const Cmplx& z) {return log(z)/=log(10);}
Cmplx polar(const Reel rho,const Reel theta) {return Cmplx(rho*Sstd::cos(theta),rho*Sstd::sin(theta));}
Cmplx pow (const Cmplx& z,Reel d) {return exp(log(z)*=d);}
Cmplx pow (const Cmplx& z,const Cmplx& x) {return exp(log(z)*=x);}
Cmplx sin (const Cmplx& z) {return Cmplx(Sstd::sin(z.R)*Sstd::cosh(z.I),Sstd::cos(z.R)*Sstd::sinh(z.I));}
Cmplx sinh(const Cmplx& z) {return Cmplx(Sstd::sinh(z.R)*Sstd::cos(z.I),Sstd::cosh(z.R)*Sstd::sin(z.I));}

Cmplx sqrt(const Cmplx& z)
{
    if (z==0) return z;
    Reel y=Sstd::sqrt(2*(abs(z)-z.R));

    if(imag(z)<std::pow(10,-16))
    {
        if(real(z)>0){return Cmplx(Sstd::sqrt(real(z)),0);}
        else{return Cmplx(0,Sstd::sqrt(-real(z)));}
    }
    return Cmplx(z.I/y,y/2);
}
Cmplx tan (const Cmplx& z) {return sin(z)/=cos(z);}
Cmplx tanh(const Cmplx& z) {return sinh(z)/=cosh(z);}

// -------------- fonctions complexes suplémentaires
Cmplx square(const Cmplx& z) {return Cmplx(z.R*z.R-z.I*z.I,2*z.R*z.I);}
Cmplx cube  (const Cmplx& z) {Reel a=z.R,aa=a*a,b=z.I,bb=b*b; return Cmplx(a*aa-3*a*bb,3*aa*b-b*bb);}
void cbrt(const Cmplx& z, Cmplx x[3]) {x[0] = exp(log(z)/3);
                x[1] = x[0]*Cmplx(-0.5,std::sqrt(0.75)); x[2] = x[0]*Cmplx(-0.5,-std::sqrt(0.75));}
