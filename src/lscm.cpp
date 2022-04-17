#include "lscm.h"
#include "vector_area_matrix.h"
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/eigs.h>
#include <Eigen/SVD>

void lscm(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  Eigen::SparseMatrix<double> A, L, Q;
  vector_area_matrix(F, A);
  igl::cotmatrix(V, F, L);
  igl::repdiag(L, 2, Q);
  Q = Q - A;

  Eigen::SparseMatrix<double> M, B;
  igl::massmatrix(V, F, igl::MassMatrixType::MASSMATRIX_TYPE_BARYCENTRIC, M);
  igl::repdiag(M, 2, B);

  // compute LSCM
  Eigen::MatrixXd sU;
  Eigen::VectorXd sS;
  igl::eigs(Q, B, 3, igl::EigsType::EIGS_TYPE_SM, sU, sS);
  // use the first non-trival solution (the 3rd one)
  // the first two eigenvectors 0 and c * I is associated with eigenvalue 0
  U = Eigen::Map<Eigen::MatrixXd>(sU.col(2).data(), V.rows(), 2);

  // compute canonical rotation 
  // use PCA to compute the axis alignment
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(U.transpose() * U, Eigen::ComputeFullU | Eigen::ComputeFullV);
  U = U * svd.matrixU();

  // compute canonical rotation (not working case)
  // // This won't exactly preserve the reflectional symmetric
  // // Because U_mean is not necessarily at (0, 0)
  // Eigen::RowVectorXd U_mean = U.colwise().mean();
  // Eigen::MatrixXd U_center = U.rowwise() - U_mean;
  // std::cout << "U_mean = " << std::endl << U_mean << std::endl;
  // Eigen::JacobiSVD<Eigen::MatrixXd> svd(U_center.transpose()*U_center, Eigen::ComputeFullU | Eigen::ComputeFullV);
  // Eigen::MatrixXd R = svd.matrixU();
  // U = (U_center * R.transpose()).rowwise() + U_mean;
}
