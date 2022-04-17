#include "vector_area_matrix.h"
#include <igl/boundary_loop.h>
#include <igl/min_quad_with_fixed.h>
#include <vector>

void vector_area_matrix(
  const Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double>& A)
{
  int V_size = F.maxCoeff()+1;
  A.resize(V_size*2, V_size*2);
  A.setZero();

  // compute boundary loop
  std::vector<std::vector<int>> bi;
  igl::boundary_loop(F, bi);

  // loop over boundary
  std::vector<Eigen::Triplet<double>> triplets;
  int i, j;
  for(int l = 0; l < bi.size(); l++) {
    for(int k = 0; k < bi[l].size(); k++) {
      i = bi[l][k];
      j = bi[l][(k < bi[l].size()-1) ? k+1:0];
      triplets.push_back(Eigen::Triplet<double>(i, V_size + j, 0.5));
      triplets.push_back(Eigen::Triplet<double>(V_size +  i, j, -0.5));
      // symmetric
      triplets.push_back(Eigen::Triplet<double>(V_size + j, i,  0.5));
      triplets.push_back(Eigen::Triplet<double>(j, V_size +  i, -0.5));
    }
  }

  A.setFromTriplets(triplets.begin(), triplets.end());
}

