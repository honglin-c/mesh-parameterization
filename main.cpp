#include "tutte.h"
#include "lscm.h"
#include <igl/cat.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/slim.h>
#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/writeOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <string>
#include <iostream>

int main(int argc, char *argv[])
{
  // Load input meshes
  Eigen::MatrixXd V,U_lscm,U_tutte,U_arap,U_slim,U;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(
    (argc>1?argv[1]:"../data/beetle.obj"),V,F);
//   igl::opengl::glfw::Viewer viewer;
//   std::cout<<R"(
// [space]  Toggle whether displaying 3D surface or 2D parameterization
// C,c      Toggle checkerboard
// t        Switch parameterization to Tutte embedding
// l        Switch parameterization to Least squares conformal mapping
// a        Switch parameterization to ARAP parameterization
// s        Switch parameterization to SLIM parameterization
// )";
  tutte(V,F,U_tutte);
  lscm(V,F,U_lscm);

  // Compute the initial solution for ARAP (harmonic parametrization) and SLIM
  Eigen::MatrixXd initial_guess;
  Eigen::VectorXi bnd;
  igl::boundary_loop(F,bnd);
  Eigen::MatrixXd bnd_uv;
  igl::map_vertices_to_circle(V,bnd,bnd_uv);
  igl::harmonic(V,F,bnd,bnd_uv,1,initial_guess);
  Eigen::VectorXi b  = Eigen::VectorXi::Zero(0);
  Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0,0);

  // ARAP parameterization
  {
    // Add dynamic regularization to avoid to specify boundary conditions
    igl::ARAPData arap_data;
    arap_data.with_dynamics = true;

    // Initialize ARAP
    arap_data.max_iter = 100;
    // 2 means that we're going to *solve* in 2d
    arap_precomputation(V,F,2,b,arap_data);

    // Solve arap using the harmonic map as initial guess
    U_arap = initial_guess;
    arap_solve(bc,arap_data,U_arap);

    // Scale UV to make the texture more clear
    U_arap *= 20;

    // flip an axis to make it look better
    U_arap.col(1) *= -1; 
  }

  // SLIM parameterization
  {
    igl::MappingEnergyType slim_energy = igl::SYMMETRIC_DIRICHLET;
    igl::SLIMData slim_data;
    Eigen::MatrixXd V_init = initial_guess;
    double soft_p = 0;
    slim_precompute(V,F,V_init,slim_data,slim_energy,b,bc,soft_p);
    double slim_max_iter = 50;
    U_slim = slim_solve(slim_data, slim_max_iter);
    
    // flip an axis to make it look better
    U_slim.col(1) *= -1; 
  }
  // Fit parameterization in unit sphere
  const auto normalize = [](Eigen::MatrixXd &U)
  {
    U.rowwise() -= U.colwise().mean().eval();
    U.array() /= 
      (U.colwise().maxCoeff() - U.colwise().minCoeff()).maxCoeff()/2.0;
  };
  normalize(V);
  normalize(U_tutte);
  normalize(U_lscm);
  normalize(U_arap);
  normalize(U_slim);

  // output the embeddings
  Eigen::MatrixXd col_zero = Eigen::MatrixXd::Zero(V.rows(),1);
  Eigen::MatrixXd V_tutte, V_lscm, V_arap, V_slim;
  V_tutte = igl::cat(2, U_tutte, col_zero);
  V_lscm = igl::cat(2, U_lscm, col_zero);
  V_arap = igl::cat(2, U_arap, col_zero);
  V_slim = igl::cat(2, U_slim, col_zero);

  igl::writeOBJ("../output/original.obj", V, F);
  igl::writeOBJ("../output/tutte.obj", V_tutte, F);
  igl::writeOBJ("../output/lscm.obj", V_lscm, F);
  igl::writeOBJ("../output/arap.obj", V_arap, F);
  igl::writeOBJ("../output/slim.obj", V_slim, F);

  // // Compute per-face distortion
  // int nfaces = F.rows();
  // face_colors.resize(nfaces, 3);
  // for(int i=0; i<nfaces; i++)
  // {
  //   Eigen::Matrix<double, 2, 3> M1;
  //   M1.row(0) = V.row(F(i,1)) - V.row(F(i,0));
  //   M1.row(1) = V.row(F(i,2)) - V.row(F(i,0));
  //   Eigen::Matrix2d M2;
  //   M2.row(0) = V_uv.row(F(i,1)) - V_uv.row(F(i,0));
  //   M2.row(1) = V_uv.row(F(i,2)) - V_uv.row(F(i,0));
  //   Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  //   M -= (M1*M1.transpose()).inverse() * M2*M2.transpose();
  //   Eigen::Vector2cd evals = M.eigenvalues();
    
  //   double lmax = std::max(std::real(evals[0]), std::real(evals[1]));
  //   double lmin = std::min(std::real(evals[0]), std::real(evals[1]));
  //   double magic = 0.5;
  //   face_colors(i,0) = 1.0 - magic*std::max(0.0, -lmin);
  //   face_colors(i,1) = 1.0 - magic*std::max(lmax, -lmin);
  //   face_colors(i,2) = 1.0 - magic*std::max(0.0, lmax);
  // }


  // bool plot_parameterization = false;
  // const auto & update = [&]()
  // {
  //   if(plot_parameterization)
  //   {
  //     // Viewer wants 3D coordinates, so pad UVs with column of zeros
  //     viewer.data().set_vertices(
  //       (Eigen::MatrixXd(V.rows(),3)<<
  //        U.col(0),Eigen::VectorXd::Zero(V.rows()),U.col(1)).finished());
  //   }else
  //   {
  //     viewer.data().set_vertices(V);
  //   }
  //   viewer.data().compute_normals();
  //   viewer.data().set_uv(U*10);
  // };
  // viewer.callback_key_pressed = 
  //   [&](igl::opengl::glfw::Viewer &, unsigned int key, int)
  // {
  //   switch(key)
  //   {
  //     case ' ':
  //       plot_parameterization ^= 1;
  //       break;
  //     case 'L':
  //     case 'l':
  //       U = U_lscm;
  //       break;
  //     case 'T':
  //     case 't':
  //       U = U_tutte;
  //       break;
  //     case 'A':
  //     case 'a':
  //       U = U_arap;
  //       break;
  //     case 'S':
  //     case 's':
  //       U = U_slim;
  //       break;
  //     case 'C':
  //     case 'c':
  //       viewer.data().show_texture ^= 1;
  //       break;
  //     default:
  //       return false;
  //   }
  //   update();
  //   return true;
  // };

  // U = U_tutte;
  // viewer.data().set_mesh(V,F);
  // Eigen::MatrixXd N;
  // igl::per_vertex_normals(V,F,N);
  // viewer.data().set_colors(N.array()*0.5+0.5);
  // update();
  // viewer.data().show_texture = true;
  // viewer.data().show_lines = false;
  // viewer.launch();

  return EXIT_SUCCESS;
}
