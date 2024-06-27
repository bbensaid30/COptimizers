#include <iostream>
#include "shaman.h"

#include "Cmplx.h"

void ResolER3(Sdouble const& a, Sdouble const& b, Sdouble const& c, Sdouble const& d, Cmplx x[3]);
Sdouble solPositive(Sdouble const& a, Sdouble const& b, Sdouble const& c, Sdouble const& d);
Sdouble pasAdaptatifMomentum(Sdouble const& beta_bar, Sdouble const& deltaR, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar);
Sdouble deltah(Sdouble const& beta_bar, Sdouble const& deltaR, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar, Sdouble const& h);
Sdouble solMomentum(Sdouble const& beta_bar, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar, Sdouble const& h, Sdouble const& delta);
Sdouble* solsMomentum(Sdouble const& beta_bar, Sdouble const& grad2R, Sdouble const& norm2v, Sdouble const& prodScalar, Sdouble const& h, Sdouble const& delta);
