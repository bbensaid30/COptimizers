#ifndef PERTE
#define PERTE

#include <string>
#include <cmath>
#include <cassert>

#include <Eigen/Dense>

//--------------------------------------------------------------- Calcul de L -----------------------------------------------------------------------------------------------

double norme2(Eigen::VectorXd const& x, Eigen::VectorXd const& y);
double difference(Eigen::VectorXd const& x, Eigen::VectorXd const& y);
double entropie_generale(Eigen::VectorXd const& x, Eigen::VectorXd const& y);
double entropie_one(Eigen::VectorXd const& x, Eigen::VectorXd const& y);
double KL_divergence(Eigen::VectorXd const& x, Eigen::VectorXd const& y);

double L(Eigen::VectorXd const& x, Eigen::VectorXd const& y, std::string type_perte="norme2");

//----------------------------------------------------------------- Calcul de L' --------------------------------------------------------------------------------------------

void FO_norme2(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP);
void FO_difference(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP);
void FO_entropie_generale(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP);
void FO_entropie_one(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP);
void FO_KL_divergence(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP);

void FO_L(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, std::string type_perte="norme2");

//--------------------------------------------------------------- Calcul L' et L'' -------------------------------------------------------------------------------------------

void SO_norme2(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP);
void SO_difference(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP);
void SO_entropie_generale(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP);
void SO_entropie_one(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP);
void SO_KL_divergence(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP);

void SO_L(Eigen::VectorXd const& x, Eigen::VectorXd const& y, Eigen::VectorXd& LP, Eigen::MatrixXd& LPP, std::string type_perte="norme2");

#endif
