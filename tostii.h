/**
 * Timestepping extracted from deal.ii to give it a better interface, faster
 * compile / development loop
 */
#pragma once
#ifndef TOSTII_HPP
#define TOSTII_HPP

#include <complex>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) &&     \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <cmath>
#include <functional>
#include <vector>

namespace tostii{

/**
 * Abstract class for time stepping methods. These methods assume that the
 * equation has the form: $ \frac{\partial y}{\partial t} = f(t,y) $.
 */
template <typename VectorType, typename TimeType = double> class TimeStepping {
public:
  /**
   * Virtual destructor.
   */
  virtual ~TimeStepping() = default;

  /**
   * Purely virtual function. This function is used to advance from time @p
   * t to t+ @p delta_t. @p F is a vector of functions $ f(t,y) $ that
   * should be integrated, the input parameters are the time t and the
   * vector y and the output is value of f at this point. @p J_inverse is a
   * vector functions that compute the inverse of the Jacobians associated
   * to the implicit problems. The input parameters are the time, $ \tau $,
   * and a vector. The output is the value of function at this point. This
   * function returns the time at the end of the time step.
   */
  virtual TimeType
  evolve_one_time_step(std::vector<std::function<void(const TimeType,      //
                                                      const VectorType&, //
                                                      VectorType&)>>& F,
                       std::vector<std::function<void(const TimeType,      //
                                                      const TimeType,      //
                                                      const VectorType&, //
                                                      VectorType&)>>& J_inverse,
                       TimeType t, TimeType delta_t, VectorType& y) = 0;

  /**
   * Empty structure used to store information.
   */
  struct Status {};

  /**
   * Purely virtual function that return Status.
   */
  virtual const Status& get_status() const = 0;
};

/** **************************************************************************

    Explicit RK

*/
enum runge_kutta_method {

  // --------------------------------------------------------------------------
  //  Explicit RK

  /**
   * Forward Euler method, first order.
   */
  FORWARD_EULER,
  /**
   * Explicit midpoint method, 2nd order.
   */
  EXPLICIT_MIDPOINT,
  /**
   * Heun method, 2nd order.
   */
  HEUN2,
  /**
   * Third order Runge-Kutta method.
   */
  RK_THIRD_ORDER,
  /**
   * Third order Strong Stability Preserving (SSP) Runge-Kutta method
   * (SSP time discretizations are also called Total Variation Diminishing
   * (TVD) methods in the literature, see @cite gottlieb2001strong).
   */
  SSP_THIRD_ORDER,
  /**
   * Classical fourth order Runge-Kutta method.
   */
  RK_CLASSIC_FOURTH_ORDER,

  // --------------------------------------------------------------------------
  //  Implicit RK

  /**
   * Backward Euler method, first order.
   */
  BACKWARD_EULER,
  /**
   * Implicit midpoint method, second order.
   */
  IMPLICIT_MIDPOINT,
  /**
   * Crank-Nicolson method, second order.
   */
  CRANK_NICOLSON,
  /**
   * Two stage SDIRK method (short for "singly diagonally implicit
   * Runge-Kutta"), second order.
   */
  SDIRK_TWO_STAGES,
  /**
   * Three stage L-stable SDIRK method, third order.
   */
  SDIRK_THREE_STAGES,
  /**
   * Three stage A-stable SDIRK method, fourth order.
   */
  SDIRK_3O4,
  /**
   * Three stage A-stable SDIRK method, fourth order.
   */
  SDIRK_5O4,

  INVALID
};

/**
 * ExplicitRungeKutta is derived from RungeKutta and implement the explicit
 * methods.
 */
template <typename VectorType, typename TimeType=double>
class ExplicitRungeKutta : public TimeStepping<VectorType,TimeType> {
public:
  /**
   * Default constructor. This constructor creates an object for which
   * you will want to call <code>initialize(runge_kutta_method)</code>
   * before it can be used.
   */
  ExplicitRungeKutta() = default;

  /**
   * Constructor. This function calls initialize(runge_kutta_method).
   */
  ExplicitRungeKutta(const runge_kutta_method method);

  /**
   * Initialize the explicit Runge-Kutta method.
   */
  void initialize(const runge_kutta_method method);

  TimeType evolve_one_time_step(
      std::vector<
          std::function<void(const TimeType, const VectorType&, VectorType&)>>& F,
      std::vector<std::function<void(const TimeType, const TimeType,
                                     const VectorType&, VectorType&)>>&
             J_inverse,
      TimeType t, TimeType delta_t, VectorType& y);

  /**
   * This function is used to advance from time @p t to t+ @p delta_t. @p f
   * is the function $ f(t,y) $ that should be integrated, the input
   * parameters are the time t and the vector y and the output is value of f
   * at this point. @p id_minus_tau_J_inverse is a function that computes $
   * inv(I-\tau J)$ where $ I $ is the identity matrix, $ \tau $ is given,
   * and $ J $ is the Jacobian $ \frac{\partial f}{\partial y} $. The input
   * parameter are the time, $ \tau $, and a vector. The output is the value
   * of function at this point. evolve_one_time_step returns the time at the
   * end of the time step.
   *
   * @note @p id_minus_tau_J_inverse is ignored since the method is explicit.
   */
  TimeType evolve_one_time_step(
      const std::function<void(const TimeType, const VectorType&, VectorType&)>&
                                              f,
      const std::function<void(const TimeType, const TimeType, const VectorType&,
                               VectorType&)>& id_minus_tau_J_inverse,
      TimeType t, TimeType delta_t, VectorType& y);

  /**
   * This function is used to advance from time @p t to t+ @p delta_t. This
   * function is similar to the one derived from RungeKutta, but does not
   * required id_minus_tau_J_inverse because it is not used for explicit
   * methods. evolve_one_time_step returns the time at the end of the time
   * step.
   */
  TimeType evolve_one_time_step(
      const std::function<void(const TimeType, const VectorType&, VectorType&)>&
             f,
      TimeType t, TimeType delta_t, VectorType& y);

  /**
   * This structure stores the name of the method used.
   */
  /**
   * This structure stores the name of the method used.
   */
  struct Status : public TimeStepping<VectorType,TimeType>::Status {
    Status() : method(INVALID) {}

    runge_kutta_method method;
  };

  /**
   * Return the status of the current object.
   */
  const Status& get_status() const;

private:
  /**
   * Compute the different stages needed.
   */
  void compute_stages(const std::function<void(const TimeType, const VectorType&,
                                               VectorType&)>& f,
                      const TimeType t, const TimeType delta_t, const VectorType& y,
                      std::vector<VectorType>& f_stages) const;

  /**
   * Number of stages of the Runge-Kutta method.
   */
  unsigned int n_stages;

  /**
   * Butcher tableau coefficients.
   */
  std::vector<double> b;

  /**
   * Butcher tableau coefficients.
   */
  std::vector<double> c;

  /**
   * Butcher tableau coefficients.
   */
  std::vector<std::vector<double>> a;

  /**
   * Status structure of the object.
   */
  Status status;
};

/** **************************************************************************

    Implicit RK

*/

/**
 * This class is derived from RungeKutta and implement the implicit methods.
 * This class works only for Diagonal Implicit Runge-Kutta (DIRK) methods.
 */
template <typename VectorType, typename TimeType=double>
class ImplicitRungeKutta : public TimeStepping<VectorType, TimeType> {
public:
  /**
   * Default constructor. initialize(runge_kutta_method) and
   * set_newton_solver_parameters(unsigned int,double) need to be called
   * before the object can be used.
   */
  ImplicitRungeKutta() = default;

  /**
   * Constructor. This function calls initialize(runge_kutta_method)
   * and initialize the maximum number of iterations and the tolerance of the
   * Newton solver.
   */
  ImplicitRungeKutta(const runge_kutta_method method,
                     const unsigned int       max_it    = 100,
                     const double             tolerance = 1e-6);

  /**
   * Initialize the implicit Runge-Kutta method.
   */
  void initialize(const runge_kutta_method method);

  TimeType evolve_one_time_step(
      std::vector<
          std::function<void(const TimeType, const VectorType&, VectorType&)>>& F,
      std::vector<std::function<void(const TimeType, const TimeType,
                                     const VectorType&, VectorType&)>>&
             J_inverse,
      TimeType t, TimeType delta_t, VectorType& y);

  /**
   * This function is used to advance from time @p t to t+ @p delta_t. @p f
   * is the function $ f(t,y) $ that should be integrated, the input
   * parameters are the time t and the vector y and the output is value of f
   * at this point. @p id_minus_tau_J_inverse is a function that computes $
   * (I-\tau J)^{-1}$ where $ I $ is the identity matrix, $ \tau $ is given,
   * and $ J $ is the Jacobian $ \frac{\partial f}{\partial y} $. The input
   * parameters this function receives are the time, $ \tau $, and a vector.
   * The output is the value of function at this point. evolve_one_time_step
   * returns the time at the end of the time step.
   */
  TimeType evolve_one_time_step(
      const std::function<void(const TimeType, const VectorType&, VectorType&)>&
                                              f,
      const std::function<void(const TimeType, const TimeType, const VectorType&,
                               VectorType&)>& id_minus_tau_J_inverse,
      TimeType t, TimeType delta_t, VectorType& y);

  /**
   * Set the maximum number of iterations and the tolerance used by the
   * Newton solver.
   */
  void set_newton_solver_parameters(const unsigned int max_it,
                                    const double       tolerance);

  /**
   * Structure that stores the name of the method, the number of Newton
   * iterations and the norm of the residual when exiting the Newton solver.
   */
  struct Status : public TimeStepping<VectorType,TimeType>::Status {
    Status() : method(INVALID), n_iterations(10), norm_residual() {}

    runge_kutta_method method;
    unsigned int       n_iterations;
    double             norm_residual;
  };

  /**
   * Return the status of the current object.
   */
  const Status& get_status() const;

private:
  /**
   * Compute the different stages needed.
   */
  void compute_stages(
      const std::function<void(const TimeType, const VectorType&, VectorType&)>&
                                              f,
      const std::function<void(const TimeType, const TimeType, const VectorType&,
                               VectorType&)>& id_minus_tau_J_inverse,
      TimeType t, TimeType delta_t, VectorType& y,
      std::vector<VectorType>& f_stages);

  /**
   * Newton solver used for the implicit stages.
   */
  void newton_solve(
      const std::function<void(const VectorType&, VectorType&)>& get_residual,
      const std::function<void(const VectorType&, VectorType&)>&
                  id_minus_tau_J_inverse,
      VectorType& y);

  /**
   * Compute the residual needed by the Newton solver.
   */
  void compute_residual(
      const std::function<void(const TimeType, const VectorType&, VectorType&)>&
             f,
      TimeType t, TimeType delta_t, const VectorType& new_y, const VectorType& y,
      VectorType& tendency, VectorType& residual) const;

  /**
   * Number of stages of the Runge-Kutta method.
   */
  unsigned int n_stages;

  /**
   * Butcher tableau coefficients.
   */
  std::vector<double> b;

  /**
   * Butcher tableau coefficients.
   */
  std::vector<double> c;

  /**
   * Butcher tableau coefficients.
   */
  std::vector<std::vector<double>> a;

  /**
   * Maximum number of iterations of the Newton solver.
   */
  unsigned int max_it;

  /**
   * Tolerance of the Newton solver.
   */
  double tolerance;

  /**
   * Status structure of the object.
   */
  Status status;
};

  /**
   * Exact time integration class.
   *
   */
  template <typename VectorType, typename TimeType=double>
  class Exact final : public TimeStepping<VectorType, TimeType>
  {
  public:
    Exact() = default;
    ~Exact() override = default;

    /**
     * This function is used to advance from time @p t to t+ @p delta_t. @p
     * nofunc is not used in this routine. @p exact_eval is a function that
     * evaluates the exact solution at time t + delta_t. @p y is state at time t
     * on input and at time t+delta_t on output.
     *
     * evolve_one_time_step returns the time at the end of the time step.
     */
    TimeType evolve_one_time_step(
        std::vector<std::function<void(const TimeType,      //
                                       const VectorType&, //
                                       VectorType&)>>& f,
        std::vector<std::function<void(const TimeType,      //
                                       const TimeType,      //
                                       const VectorType&, //
                                       VectorType&)>>& id_minus_tau_J_inverse,
        TimeType t, TimeType delta_t, VectorType& y);

    /**
     *
     */
    struct Status : public TimeStepping<VectorType,TimeType>::Status
    {
      Status() = default;
    };

    /**
     * Return the status of the current object.
     */
    const Status &
    get_status() const override;

    /**
     * Status structure of the object.
     */
  private:
    Status status;
  };


/** **************************************************************************

    OperatorSplit

*/
/*
  Pair for encoding an operator split stage:
  - op_num = which operator
  - alpha = alpha value for this stage
*/
  template <typename TimeType> struct OSpair {
    size_t   op_num;
    TimeType alpha;
  };
  typedef std::vector<int> OSmask;

  template <typename VectorType, typename TimeType = double> struct OSoperator {
    // Function signature types
    using f_fun_type =
        std::function<void(const TimeType, const VectorType&, VectorType&)>;
    using jac_fun_type = std::function<void(const TimeType, const TimeType,
                                            const VectorType&, VectorType&)>;
    // Data
    TimeStepping<VectorType, TimeType>* method;
    f_fun_type                          function;
    jac_fun_type                        id_minus_tau_J_inverse;
  };

/**
 * Class for OperatorSplit time stepping
 * - BlockVectors for multiple components
 */
template <typename BVectorType, typename TimeType=double>
class OperatorSplit : public TimeStepping<BVectorType,TimeType> {
public:
  // Function signature types
  using f_fun_type =
      std::function<void(const TimeType, const BVectorType&, BVectorType&)>;
  using f_vfun_type   = std::vector<f_fun_type>;
  using jac_fun_type  = std::function<void(const TimeType, const TimeType,
                                          const BVectorType&, BVectorType&)>;
  using jac_vfun_type = std::vector<jac_fun_type>;

  /**
   * Constructors.
   */
  // For named OS methods
  OperatorSplit(const std::string                                    method, //
                const std::vector<OSoperator<BVectorType, TimeType>> operators);

  OperatorSplit(
      const std::string                                    method,    //
      const std::vector<OSoperator<BVectorType, TimeType>> operators, //
      const std::vector<OSmask>                            mask,      //
      const BVectorType                                    ref);

  // For stages specified explicitly
  OperatorSplit(
      const std::vector<OSoperator<BVectorType, TimeType>> operators, //
      const std::vector<OSpair<TimeType>>                  stages);

  OperatorSplit(
      const BVectorType                                    ref,       //
      const std::vector<OSoperator<BVectorType, TimeType>> operators, //
      const std::vector<OSpair<TimeType>>                  stages,    //
      const std::vector<OSmask>                            mask);

  /**
   * Destructor.
   */
  ~OperatorSplit() = default;

  /**
   * This function is used to advance from time @p t to t+ @p delta_t.
   * @p f is a vector of functions $ f_j(t,y) $ that should be
   * integrated, the input parameters are the time t and the vector y
   * and the output is value of f at this point.
   * @p id_minus_tau_J_inverse is a vector of functions that computes
   * $(I-\tau * J_j)^{-1}$ where $ I $ is the identity matrix, $ \tau $ is given,
   * and $ J_j $ is the Jacobian $ \frac{\partial f_j}{\partial y_j}
   * This function passes the f_j and thier corresponding I-J_j^{-1} functions
   * through to the sub-integrators that were setup on construction.
   */
  TimeType evolve_one_time_step(f_vfun_type&   f,
                              jac_vfun_type& id_minus_tau_J_inverse, TimeType t,
                              TimeType delta_t, BVectorType& y);

  /** Trimmed down function for an OS method that has been setup */
  TimeType evolve_one_time_step(TimeType t, TimeType delta_t, BVectorType& y);

  /**
   *
   */
  struct Status : public TimeStepping<BVectorType, TimeType>::Status {
    Status() = default;
  };

  /**
   * Return the status of the current object.
   */
  const Status& get_status() const;

private:
  /*
    Operator substeppers and their stage orderings
  */
  // reference vector for global state properties
  BVectorType ref_state;
  // operator definitions
  std::vector<OSoperator<BVectorType, TimeType>> operators;
  // operator splitting stages
  std::vector<OSpair<TimeType>> stages;
  // which blocks participate in which operators
  std::vector<OSmask> mask;
  // number of block vectors in each state
  std::vector<int> nblocks;
  // vector of references to each operators needed blocks
  std::vector<BVectorType> blockrefs;
  // name of the method being employed (empty string "" if specified by stages)
  std::string method_name;

  Status status;

  // validate that the method name provided to the class is correct
  // (exception thrown if not)
  void check_method_name(std::string method_name);
};


// Coeffs for PP_3_A_3
static const double a_pp3a3[3] = {0.461601939364879971,   //
                                  -0.0678710530507800810, //
                                  -0.0958868852260720250};
static const double b_pp3a3[3] = {-0.266589223588183997, //
                                  0.0924576733143338350, //
                                  0.674131550273850162};
static const double c_pp3a3[3] = {-0.360420727960349671, //
                                  0.579154058410941403,  //
                                  0.483422668461380403};

static const double y_omega[2] = {-1.702414383919315268, //
                                  1.351207191959657634};

/**
   Define a small dictionary of possible OS methods
*/
static const std::unordered_map<std::string, std::vector<OSpair<double>>>
    os_method{
        {"Godunov", std::vector<OSpair<double>>{OSpair<double>{0, 1.0},      //
                                                OSpair<double>{1, 1.0}}},    //
        {"Strang", std::vector<OSpair<double>>{OSpair<double>{0, 0.5},       //
                                               OSpair<double>{1, 1.0},       //
                                               OSpair<double>{0, 0.5}}},     //
        {"Ruth", std::vector<OSpair<double>>{OSpair<double>{0, 7.0 / 24.0},  //
                                             OSpair<double>{1, 2.0 / 3.0},   //
                                             OSpair<double>{0, 3.0 / 4.0},   //
                                             OSpair<double>{1, -2.0 / 3.0},  //
                                             OSpair<double>{0, -1.0 / 24.0}, //
                                             OSpair<double>{1, 1.0}}},       //
        {"Yoshida",
         std::vector<OSpair<double>>{
             OSpair<double>{0, y_omega[1] / 2.0},              //
             OSpair<double>{1, y_omega[1]},                    //
             OSpair<double>{0, (y_omega[0] + y_omega[1]) / 2.0}, //
             OSpair<double>{1, y_omega[0]},                    //
             OSpair<double>{0, (y_omega[0] + y_omega[1]) / 2.0}, //
             OSpair<double>{1, y_omega[1]},                    //
             OSpair<double>{0, y_omega[1] / 2.0}}},            //

        // 3-split methods
        {"Godunov3", std::vector<OSpair<double>>{OSpair<double>{0, 1.0},   //
                                                 OSpair<double>{1, 1.0},   //
                                                 OSpair<double>{2, 1.0}}}, //
        {"Strang3", std::vector<OSpair<double>>{OSpair<double>{0, 0.5},    //
                                                OSpair<double>{1, 0.5},    //
                                                OSpair<double>{2, 1.0},    //
                                                OSpair<double>{1, 0.5},    //
                                                OSpair<double>{0, 0.5}}},  //
        {"PP_3_A_3",
         std::vector<OSpair<double>>{OSpair<double>{0, a_pp3a3[0]},  //
                                     OSpair<double>{1, b_pp3a3[0]},  //
                                     OSpair<double>{2, c_pp3a3[0]},  //
                                     OSpair<double>{0, a_pp3a3[1]},  //
                                     OSpair<double>{1, b_pp3a3[1]},  //
                                     OSpair<double>{2, c_pp3a3[1]},  //
                                     OSpair<double>{0, a_pp3a3[2]},  //
                                     OSpair<double>{1, b_pp3a3[2]},  //
                                     OSpair<double>{2, c_pp3a3[2]},  //
                                     OSpair<double>{0, c_pp3a3[2]},  //
                                     OSpair<double>{1, b_pp3a3[2]},  //
                                     OSpair<double>{2, a_pp3a3[2]},  //
                                     OSpair<double>{0, c_pp3a3[1]},  //
                                     OSpair<double>{1, b_pp3a3[1]},  //
                                     OSpair<double>{2, a_pp3a3[1]},  //
                                     OSpair<double>{0, c_pp3a3[0]},  //
                                     OSpair<double>{1, b_pp3a3[0]},  //
                                     OSpair<double>{2, a_pp3a3[0]}}} //
    };

static const std::complex<double> i = {0, 1};

// coefficients for PP_3_A_3_c method
static const std::complex<double> a_pp3a3c[3] =
    {0.0442100822731214750 - 0.0713885293035937610 * i,
     0.157419072651724312 - 0.1552628290245811054 * i,
     0.260637333463417766 + 0.07744172526769638060 * i};
static const std::complex<double> b_pp3a3c[3] = {
    0.0973753110633760580 - 0.112390152630243038 * i,
    0.179226865237094561 - 0.0934263750859694960 * i,
    0.223397823699529381 + 0.205816527716212534 * i};
static const std::complex<double> c_pp3a3c[3] = {
    0.125415464915697242 - 0.281916718734615225 * i,
    0.353043498499040389 + 0.0768951336684972038 * i,
    0.059274548196998816 + 0.354231218126596507 * i};

/**
   Define a small dictionary of possible complex OS methods
*/
static const std::unordered_map<std::string, //
                                std::vector<OSpair<std::complex<double>>>>
    os_complex{
        {"Milne_2_2_c_i",
         std::vector<OSpair<std::complex<double>>>{
             OSpair<std::complex<double>>{0, 12.0 / 37.0 - 2.0 / 37.0 * i},  //
             OSpair<std::complex<double>>{1, 25.0 / 34.0 - 1.0 / 17.0 * i},  //
             OSpair<std::complex<double>>{0, 25.0 / 37.0 + 2.0 / 37.0 * i},  //
             OSpair<std::complex<double>>{1, 9.0 / 34.0 + 1.0 / 17.0 * i}}}, //

        {"Milne_2_2_c_i_asc",
         std::vector<OSpair<std::complex<double>>>{
             OSpair<std::complex<double>>{0, 0.8 - 0.4 * i}, //
             OSpair<std::complex<double>>{1, 0.5 + i},       //
             OSpair<std::complex<double>>{0, 0.2 + 0.4 * i}, //
             OSpair<std::complex<double>>{1, 0.5 - i}}},     //
        {"Milne_2_2_c_ii",
         std::vector<OSpair<std::complex<double>>>{
             OSpair<std::complex<double>>{0, 4. / 13. - 1. / 26. * i},   //
             OSpair<std::complex<double>>{1, 18. / 25. - 1. / 25. * i},  //
             OSpair<std::complex<double>>{0, 9. / 13. + 1. / 26. * i},   //
             OSpair<std::complex<double>>{1, 7. / 25. + 1. / 25. * i}}}, //
        {"A_3_3_c",
         std::vector<OSpair<std::complex<double>>>{
             OSpair<std::complex<double>>{1,
                                          0.25 + 0.144337567297406441 * i},   //
             OSpair<std::complex<double>>{0, 0.5 + 0.288675134594812882 * i}, //
             OSpair<std::complex<double>>{1, 0.5},                            //
             OSpair<std::complex<double>>{0, 0.5 - 0.288675134594812882 * i}, //
             OSpair<std::complex<double>>{1, 0.25 - 0.144337567297406441 * i}}},
        // 3-split methods
        {"AKT_2_2_c",
         std::vector<OSpair<std::complex<double>>>{
             OSpair<std::complex<double>>{0, 0.5 + 0.5 * i},   //
             OSpair<std::complex<double>>{1, 0.5 + 0.5 * i},   //
             OSpair<std::complex<double>>{2, 0.5 + 0.5 * i},   //
             OSpair<std::complex<double>>{0, 0.5 - 0.5 * i},   //
             OSpair<std::complex<double>>{1, 0.5 - 0.5 * i},   //
             OSpair<std::complex<double>>{2, 0.5 - 0.5 * i}}}, //
        {"PP_3_A_3_c",
         std::vector<OSpair<std::complex<double>>>{
             OSpair<std::complex<double>>{0, a_pp3a3c[0]},  //
             OSpair<std::complex<double>>{1, b_pp3a3c[0]},  //
             OSpair<std::complex<double>>{2, c_pp3a3c[0]},  //
             OSpair<std::complex<double>>{0, a_pp3a3c[1]},  //
             OSpair<std::complex<double>>{1, b_pp3a3c[1]},  //
             OSpair<std::complex<double>>{2, c_pp3a3c[1]},  //
             OSpair<std::complex<double>>{0, a_pp3a3c[2]},  //
             OSpair<std::complex<double>>{1, b_pp3a3c[2]},  //
             OSpair<std::complex<double>>{2, c_pp3a3c[2]},  //
             OSpair<std::complex<double>>{0, c_pp3a3c[2]},  //
             OSpair<std::complex<double>>{1, b_pp3a3c[2]},  //
             OSpair<std::complex<double>>{2, a_pp3a3c[2]},  //
             OSpair<std::complex<double>>{0, c_pp3a3c[1]},  //
             OSpair<std::complex<double>>{1, b_pp3a3c[1]},  //
             OSpair<std::complex<double>>{2, a_pp3a3c[1]},  //
             OSpair<std::complex<double>>{0, c_pp3a3c[0]},  //
             OSpair<std::complex<double>>{1, b_pp3a3c[0]},  //
             OSpair<std::complex<double>>{2, a_pp3a3c[0]}}} //
    };

/**
 * Class for OperatorSplitSingle time stepping
 * - single component problem
 */
template <typename VectorType, typename TimeType = double>
class OperatorSplitSingle : public TimeStepping<VectorType, TimeType> {
public:
  // Function signature types
  using f_fun_type =
      std::function<void(const TimeType, const VectorType&, VectorType&)>;
  using f_vfun_type   = std::vector<f_fun_type>;
  using jac_fun_type  = std::function<void(const TimeType, const TimeType,
                                          const VectorType&, VectorType&)>;
  using jac_vfun_type = std::vector<jac_fun_type>;

  /**
   * Constructors.
   */
  OperatorSplitSingle(
      const std::vector<OSoperator<VectorType, TimeType>> operators,
      const std::vector<OSpair<TimeType>>                 stages);

  OperatorSplitSingle(
      const VectorType                                    ref,
      const std::vector<OSoperator<VectorType, TimeType>> operators,
      const std::vector<OSpair<TimeType>>                 stages);

  /**
   * Destructor.
   */
  ~OperatorSplitSingle() = default;

  /**
   * This function is used to advance from time @p t to t+ @p delta_t.
   * @p f is a vector of functions $ f_j(t,y) $ that should be
   * integrated, the input parameters are the time t and the vector y
   * and the output is value of f at this point.
   * @p id_minus_tau_J_inverse is a vector of functions that computes
   * $(I-\tau * J_j)^{-1}$ where $ I $ is the identity matrix, $ \tau $ is given,
   * and $ J_j $ is the Jacobian $ \frac{\partial f_j}{\partial y_j}
   * This function passes the f_j and thier corresponding I-J_j^{-1} functions
   * through to the sub-integrators that were setup on construction.
   */
  TimeType evolve_one_time_step(f_vfun_type&   f,
                                jac_vfun_type& id_minus_tau_J_inverse,
                                TimeType t, TimeType delta_t, VectorType& y);

  /** Trimmed down function for an OS method that has been setup */
  TimeType evolve_one_time_step(TimeType t, TimeType delta_t, VectorType& y);

  /**
   * Construction of Status object is just default from the base class
   */
  struct Status : public TimeStepping<VectorType, TimeType>::Status {
    Status() = default;
  };

  /**
   * Return the status of the current object.
   */
  const Status& get_status() const;

private:
  /*
    Operator substeppers and their stage orderings
  */
  VectorType ref_state; // reference vector for global state properties
  std::vector<OSoperator<VectorType, TimeType>>
                                operators; // operator definitions
  std::vector<OSpair<TimeType>> stages;    // operator splitting stages

  Status status;
  };

  // String/ENUM manipulation
  std::string        RK_method_enum_to_string(runge_kutta_method method);
  runge_kutta_method RK_string_to_enum(std::string method_name);

}
#endif // TOSTII_HPP
