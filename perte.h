#ifndef PERTE
#define PERTE

#include <string>
#include <cmath>
#include <cassert>

#include <Eigen/Dense>
#include "shaman.h"
#include "shaman/helpers/shaman_eigen.h"

//--------------------------------------------------------------- Calcul de L -----------------------------------------------------------------------------------------------

Sdouble norme2(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y);
Sdouble difference(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y);
Sdouble entropie_generale(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y);
Sdouble entropie_one(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y);
Sdouble KL_divergence(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y);

Sdouble L(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, std::string type_perte="norme2");

//----------------------------------------------------------------- Calcul de L' --------------------------------------------------------------------------------------------

void FO_norme2(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP);
void FO_difference(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP);
void FO_entropie_generale(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP);
void FO_entropie_one(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP);
void FO_KL_divergence(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP);

void FO_L(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, std::string type_perte="norme2");

//--------------------------------------------------------------- Calcul L' et L'' -------------------------------------------------------------------------------------------

void SO_norme2(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP);
void SO_difference(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP);
void SO_entropie_generale(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP);
void SO_entropie_one(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP);
void SO_KL_divergence(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP);

void SO_L(Eigen::SVectorXd const& x, Eigen::SVectorXd const& y, Eigen::SVectorXd& LP, Eigen::SMatrixXd& LPP, std::string type_perte="norme2");

#endif
