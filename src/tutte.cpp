#include "tutte.h"
#include <igl/edges.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>

// Given a 3D mesh (`V`,`F`) with a disk topology (i.e., a manifold with single
// boundary), compute a 2D parameterization according to Tutte's mapping inside
// the unit disk. All boundary vertices should be mapped to the unit circle and
// interior vertices mapped inside the disk _without_ flips.
//
// Inputs:
//   V  #V by 3 list of mesh vertex positions
//   F  #F by 3 list of triangle indices into V
// Outputs:
//   U  #U by 2 list of mesh UV parameterization coordinates
//
void tutte(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  Eigen::SparseMatrix<double> L;
  Eigen::MatrixXi E;
  igl::edges(F, E);
  // compute graph laplacian
  L.resize(V.rows(), V.rows());
  std::vector<Eigen::Triplet<double>> triplets;
  double w;
  for(int i = 0; i < E.rows(); i++) {
    w = 1.0 / (V.row(E(i, 0))- V.row(E(i, 1))).norm();
    triplets.push_back(Eigen::Triplet<double>(E(i, 0), E(i, 1), w));
    triplets.push_back(Eigen::Triplet<double>(E(i, 1), E(i, 0), w));
    triplets.push_back(Eigen::Triplet<double>(E(i, 0), E(i, 0), -w));
    triplets.push_back(Eigen::Triplet<double>(E(i, 1), E(i, 1), -w));
  }
  L.setFromTriplets(triplets.begin(), triplets.end());
  L *= -1;

  // compute boundary loop
  Eigen::VectorXi bi;
  igl::boundary_loop(F, bi);
  // map boundary to circle
  Eigen::MatrixXd bc;
  igl::map_vertices_to_circle(V, bi, bc);
  // compute tutte embedding
  U.resize(V.rows(), 2);

  igl::min_quad_with_fixed_data<double> data;

  igl::min_quad_with_fixed_precompute(
    L, 
    bi,
    Eigen::SparseMatrix<double>(), 
    true,
    data);

  Eigen::MatrixXd B(V.rows(), 2);
  B.setZero();
  Eigen::VectorXd Beq;
  igl::min_quad_with_fixed_solve(
    data,
    B,
    bc,
    Beq,
    U);

  // flip an axis to make it look better
  U.col(1) *= -1; 
}

