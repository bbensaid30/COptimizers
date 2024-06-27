#ifndef CMPLX
#define CMPLX

#include <iostream>
#include "shaman.h"

#define Reel Sdouble

class Cmplx {
private:
  Reel R, I;

public:
  // constructeurs
  Cmplx(); // default
  Cmplx(Reel re, Reel im);
  Cmplx(const Cmplx& z);

  // affectation
  Cmplx& operator=(const Reel r);

  // affectations avec opération (Cmplx)
  Cmplx& operator += (const Cmplx& z);
  Cmplx& operator -= (const Cmplx& z);
  Cmplx& operator *= (const Cmplx& z);
  Cmplx& operator /= (const Cmplx& z);

  // affectations avec opération (Reel)
  Cmplx& operator += (const Reel d);
  Cmplx& operator -= (const Reel d);
  Cmplx& operator *= (const Reel d);
  Cmplx& operator /= (const Reel d);

  // fonctions réelles standards
  friend Reel real(const Cmplx& z);
  friend Reel imag(const Cmplx& z);
  friend Reel norm(const Cmplx& z);
  friend Reel abs(const Cmplx& z);
  friend Reel arg(const Cmplx& z);

  friend std::ostream& operator<<(std::ostream& os, const Cmplx& z);

  // opérateurs de comparaison
  friend bool operator==(const Cmplx y,const Cmplx& z);
  friend bool operator!=(const Cmplx y,const Cmplx& z);
  friend bool operator==(const Cmplx y,const Reel d);
  friend bool operator!=(const Cmplx y,const Reel d);

  // opérateurs unaires
  friend Cmplx operator+(const Cmplx& z);
  friend Cmplx operator-(const Cmplx& z);

  // opérateurs binaires
  friend Cmplx& operator+(Cmplx y,const Cmplx& z);
  friend Cmplx& operator+(const Reel a,Cmplx z);
  friend Cmplx& operator+(Cmplx z,const Reel a);
  friend Cmplx& operator-(Cmplx y,const Cmplx& z);
  friend Cmplx  operator-(const Reel a,const Cmplx& z);
  friend Cmplx& operator-(Cmplx z,const Reel a);
  friend Cmplx& operator*(Cmplx y,const Cmplx& z);
  friend Cmplx& operator*(const Reel a,Cmplx z);
  friend Cmplx& operator*(Cmplx z,const Reel a);
  friend Cmplx& operator/(Cmplx y,const Cmplx& z);
  friend Cmplx  operator/(Reel a,const Cmplx& z);
  friend Cmplx& operator/(Cmplx z,const Reel a);

  // fonctions complexes  standards
  friend Cmplx conj(const Cmplx& z);
  friend Cmplx cos (const Cmplx& z);
  friend Cmplx cosh(const Cmplx& z);
  friend Cmplx exp (const Cmplx& z);
  friend Cmplx log (const Cmplx& z);
  friend Cmplx log10(const Cmplx& z);
  friend Cmplx polar(const Reel r,const Reel t);
  friend Cmplx pow (const Cmplx& z,Reel d);
  friend Cmplx pow (const Cmplx& z,const Cmplx& x);
  friend Cmplx sin (const Cmplx& z);
  friend Cmplx sinh(const Cmplx& z);
  friend Cmplx sqrt(const Cmplx& z);
  friend Cmplx tan (const Cmplx& z);
  friend Cmplx tanh(const Cmplx& z);

  // fonctions complexes suplémentaires
  friend Cmplx square(const Cmplx& z);
  friend Cmplx cube  (const Cmplx& z);
  friend void cbrt(const Cmplx& z, Cmplx x[3]); // ³√z: x[0]³ = x[1]³ = x[2]³ = z
};

#endif
