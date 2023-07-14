
#ifndef FITZHUGHNAGUMO1961EB2008_HPP
#define FITZHUGHNAGUMO1961EB2008_HPP

// For understanding distribution of vectors
#include <deal.II/base/index_set.h>

// Note, needs to have a suitable BlockVector defined before this header is
// included:
// - LA::MPI::BlockVector

#include <cmath>

#define one_third 0.33333333333333333

struct FitzHughNagumo1961EB2008 {
  double Beta;    // [?]
  double Gamma;   //
  double Epsilon; //
  double invEps;  //

  FitzHughNagumo1961EB2008()
      : Beta(1.0),            //
        Gamma(0.5),           //
        Epsilon(0.1),         //
        invEps(1.0 / Epsilon) //
  {}
};

constexpr size_t state_size(const FitzHughNagumo1961EB2008& /*m*/) { return 2; }
constexpr size_t rate_size(const FitzHughNagumo1961EB2008& /*m*/) { return 2; }

void evaluate_y_derivatives(const FitzHughNagumo1961EB2008& m,      //
                            const LA::MPI::BlockVector&     Y,      //
                            const LA::MPI::Vector&          I_stim, //
                            LA::MPI::BlockVector&           DY) {

  auto& V = Y.block(0); // v
  auto& W = Y.block(1); // w

  auto& V_prime = DY.block(0); // v'
  auto& W_prime = DY.block(1); // w'

  /* Note: we can do this style of looping in membrane models because of the
     spatial independence of cell model rate computations. Assumes that all
     Vectors (and BlockVector.block(i)s) have the same distribution pattern.

     Adding space-dependent stimulus current will require more care in setting
     up the current values, but this loop style should still be fine here (given
     input I_stim vector).
  */
  auto local_range = V.local_range();
  for (auto i = local_range.first; i < local_range.second; ++i) {
    // dV/dt
    V_prime[i] = m.invEps * (V[i]                             //
                             - one_third * V[i] * V[i] * V[i] //
                             - W[i])                          //
                 + I_stim[i];                                 //
    // dw/dt
    W_prime[i] = m.Epsilon * (V[i]               //
                              + m.Beta           //
                              - m.Gamma * W[i]); //
  }
}

void get_I_ionic(const FitzHughNagumo1961EB2008& m, //
                 const LA::MPI::BlockVector&     Y, //
                 LA::MPI::Vector&                i_ion) {

  auto& V = Y.block(0); // v
  auto& W = Y.block(1); // w

  auto local_range = V.local_range();
  for (auto i = local_range.first; i < local_range.second; ++i) {
    i_ion[i] = m.invEps * (V[i]                             //
                           - one_third * V[i] * V[i] * V[i] //
                           - W[i]);                         //
  }
}

void equilibrium_state(const FitzHughNagumo1961EB2008& /* m */, //
                       LA::MPI::BlockVector& Y) {

  auto& V = Y.block(0); // v
  auto& W = Y.block(1); // w

  auto local_range = V.local_range();
  for (auto i = local_range.first; i < local_range.second; ++i) {
    V[i] = -1.2879118919372559; // [mV]
    W[i] = -0.5758181214332581; // [-]
  }
  return;
}

void allocate_all_cells(const FitzHughNagumo1961EB2008& m,          //
                        MPI_Comm                        comm,       //
                        dealii::IndexSet                local_dofs, //
                        LA::MPI::BlockVector&           states,     //
                        LA::MPI::BlockVector&           rates,      //
                        LA::MPI::Vector&                stimulus,   //
                        bool equilibrium_init = false) {

  /*
     Allocate block vectors for states, rates, and vector for stimulus current
  */
  states.reinit(std::vector<dealii::IndexSet>(2, local_dofs), comm);
  rates.reinit(std::vector<dealii::IndexSet>(2, local_dofs), comm);
  stimulus.reinit(local_dofs, comm);

  // Initialize to equilibrium state if desired
  if (equilibrium_init) {
    equilibrium_state(m, states);
  }
}

#endif // FITZHUGHNAGUMO1961EB2008_HPP
