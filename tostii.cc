
#include "tostii.h"

#include <fstream>

namespace tostii {

/**----------------------------------------------------------------------
 * ExplicitRungeKutta
 * ----------------------------------------------------------------------
*/

template <typename VectorType, typename TimeType>
ExplicitRungeKutta<VectorType, TimeType>::ExplicitRungeKutta(
    const runge_kutta_method method) {
  // virtual functions called in constructors and destructors never use the
  // override in a derived class
  // for clarity be explicit on which function is called
  ExplicitRungeKutta<VectorType, TimeType>::initialize(method);
}

template <typename VectorType,typename TimeType>
TimeType ExplicitRungeKutta<VectorType,TimeType>::evolve_one_time_step(
    std::vector<std::function<void(const TimeType, const VectorType&, VectorType&)>>& F,
    std::vector<std::function<void(const TimeType, const TimeType,
                                         const VectorType&, VectorType&)>>& J_inverse,
    TimeType t, TimeType delta_t, VectorType& y) {
  return evolve_one_time_step(F[0], J_inverse[0], t, delta_t, y);
}

template <typename VectorType,typename TimeType>
void ExplicitRungeKutta<VectorType,TimeType>::initialize(
    const runge_kutta_method method) {
  status.method = method;

  switch (method) {
  case (FORWARD_EULER): {
    this->n_stages = 1;
    this->b.push_back(1.0);
    this->c.push_back(0.0);

    break;
  }
  case (EXPLICIT_MIDPOINT): {
    this->n_stages = 2;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);
    this->b.push_back(0.0);
    this->b.push_back(1.0);
    this->c.push_back(0.0);
    this->c.push_back(0.5);
    std::vector<double> tmp;
    this->a.push_back(tmp);
    tmp.resize(1);
    tmp[0] = 0.5;
    this->a.push_back(tmp);

    break;
  }
  case (HEUN2): {
    this->n_stages = 2;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);
    this->b.push_back(0.5);
    this->b.push_back(0.5);
    this->c.push_back(0.0);
    this->c.push_back(1.0);
    std::vector<double> tmp;
    this->a.push_back(tmp);
    tmp.resize(1);
    tmp[0] = 1.0;
    this->a.push_back(tmp);

    break;
  }
  case (RK_THIRD_ORDER): {
    this->n_stages = 3;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);
    this->b.push_back(1.0 / 6.0);
    this->b.push_back(2.0 / 3.0);
    this->b.push_back(1.0 / 6.0);
    this->c.push_back(0.0);
    this->c.push_back(0.5);
    this->c.push_back(1.0);
    std::vector<double> tmp;
    this->a.push_back(tmp);
    tmp.resize(1);
    tmp[0] = 0.5;
    this->a.push_back(tmp);
    tmp.resize(2);
    tmp[0] = -1.0;
    tmp[1] = 2.0;
    this->a.push_back(tmp);

    break;
  }
  case (SSP_THIRD_ORDER): {
    this->n_stages = 3;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);
    this->b.push_back(1.0 / 6.0);
    this->b.push_back(1.0 / 6.0);
    this->b.push_back(2.0 / 3.0);
    this->c.push_back(0.0);
    this->c.push_back(1.0);
    this->c.push_back(0.5);
    std::vector<double> tmp;
    this->a.push_back(tmp);
    tmp.resize(1);
    tmp[0] = 1.0;
    this->a.push_back(tmp);
    tmp.resize(2);
    tmp[0] = 1.0 / 4.0;
    tmp[1] = 1.0 / 4.0;
    this->a.push_back(tmp);

    break;
  }
  case (RK_CLASSIC_FOURTH_ORDER): {
    this->n_stages = 4;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);
    std::vector<double> tmp;
    this->a.push_back(tmp);
    tmp.resize(1);
    tmp[0] = 0.5;
    this->a.push_back(tmp);
    tmp.resize(2);
    tmp[0] = 0.0;
    tmp[1] = 0.5;
    this->a.push_back(tmp);
    tmp.resize(3);
    tmp[1] = 0.0;
    tmp[2] = 1.0;
    this->a.push_back(tmp);
    this->b.push_back(1.0 / 6.0);
    this->b.push_back(1.0 / 3.0);
    this->b.push_back(1.0 / 3.0);
    this->b.push_back(1.0 / 6.0);
    this->c.push_back(0.0);
    this->c.push_back(0.5);
    this->c.push_back(0.5);
    this->c.push_back(1.0);

    break;
  }
  default: {
  }
  }
}

template <typename VectorType,typename TimeType>
TimeType ExplicitRungeKutta<VectorType,TimeType>::evolve_one_time_step(
    const std::function<void(const TimeType, const VectorType&, VectorType&)>& f,
    const std::function<
        void(const TimeType, const TimeType,
                   const VectorType&, VectorType&)>& /*id_minus_tau_J_inverse*/,
    TimeType t, TimeType delta_t, VectorType& y) {
  return evolve_one_time_step(f, t, delta_t, y);
}

template <typename VectorType,typename TimeType>
TimeType ExplicitRungeKutta<VectorType,TimeType>::evolve_one_time_step(
    const std::function<void(const TimeType, const VectorType&, VectorType&)>& f,
    TimeType t, TimeType delta_t, VectorType& y) {
  std::vector<VectorType> f_stages(this->n_stages, y);
  // Compute the different stages needed.
  compute_stages(f, t, delta_t, y, f_stages);

  // Linear combinations of the stages.
  for (unsigned int i = 0; i < this->n_stages; ++i)
    y.sadd(1., delta_t * this->b[i], f_stages[i]);

  return (t + delta_t);
}

template <typename VectorType,typename TimeType>
const typename ExplicitRungeKutta<VectorType,TimeType>::Status&
ExplicitRungeKutta<VectorType,TimeType>::get_status() const {
  return status;
}

template <typename VectorType,typename TimeType>
void ExplicitRungeKutta<VectorType, TimeType>::compute_stages(
    const std::function<void(const TimeType, const VectorType&, VectorType&)>& f,
    const TimeType t, const TimeType delta_t, const VectorType& y,
    std::vector<VectorType>& f_stages) const {
  for (unsigned int i = 0; i < this->n_stages; ++i) {
    VectorType Y(y);
    for (unsigned int j = 0; j < i; ++j)
      Y.sadd(1., delta_t * this->a[i][j], f_stages[j]);
    // Evaluate the function f at the point (t+c[i]*delta_t,Y).
    f(t + this->c[i] * delta_t, Y, f_stages[i]);
  }
}

/**----------------------------------------------------------------------
 * ImplicitRungeKutta
 * ----------------------------------------------------------------------
 */

template <typename VectorType, typename TimeType>
ImplicitRungeKutta<VectorType, TimeType>::ImplicitRungeKutta(
    const runge_kutta_method method, const unsigned int max_it,
    const double tolerance)
    : max_it(max_it), tolerance(tolerance) {
  // virtual functions called in constructors and destructors never use the
  // override in a derived class
  // for clarity be explicit on which function is called
  ImplicitRungeKutta<VectorType,TimeType>::initialize(method);
}

template <typename VectorType, typename TimeType>
TimeType ImplicitRungeKutta<VectorType, TimeType>::evolve_one_time_step(
    std::vector<std::function<void(const TimeType, const VectorType&, VectorType&)>>& F,
    std::vector<std::function<void(const TimeType, const TimeType,
                                         const VectorType&, VectorType&)>>& J_inverse,

    TimeType t, TimeType delta_t, VectorType& y) {
  return evolve_one_time_step(F[0], J_inverse[0], t, delta_t, y);
}

template <typename VectorType, typename TimeType>
void ImplicitRungeKutta<VectorType, TimeType>::initialize(
    const runge_kutta_method method) {
  status.method = method;

  switch (method) {
  case (BACKWARD_EULER): {
    this->n_stages = 1;
    this->a.push_back(std::vector<double>(1, 1.0));
    this->b.push_back(1.0);
    this->c.push_back(1.0);

    break;
  }
  case (IMPLICIT_MIDPOINT): {
    this->n_stages = 1;
    this->a.push_back(std::vector<double>(1, 0.5));
    this->b.push_back(1.0);
    this->c.push_back(0.5);

    break;
  }
  case (CRANK_NICOLSON): {
    this->n_stages = 2;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);

    this->a.push_back(std::vector<double>(1, 0.0));
    this->a.push_back(std::vector<double>(2, 0.5));

    this->b.push_back(0.5);
    this->b.push_back(0.5);

    this->c.push_back(0.0);
    this->c.push_back(1.0);

    break;
  }
  case (SDIRK_TWO_STAGES): {
    this->n_stages = 2;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);

    double const gamma = 1.0 - 1.0 / std::sqrt(2.0);

    this->b.push_back(1.0 - gamma);
    this->b.push_back(gamma);

    this->a.push_back(std::vector<double>(1, gamma));
    this->a.push_back(this->b);

    this->c.push_back(gamma);
    this->c.push_back(1.0);

    break;
  }
  case (SDIRK_THREE_STAGES): {
    this->n_stages = 3;
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);

    double const gamma = 0.4358665215;

    this->b.push_back(-3.0*gamma*gamma/2.0 + 4.0*gamma - 1./4.);
    this->b.push_back(3.0*gamma*gamma/2.0 - 5.0*gamma + 5.0/4.0 );
    this->b.push_back(gamma);

    this->a.push_back(std::vector<double>(1, gamma));
    this->a.push_back(std::vector<double>{0.5*(1.0-gamma), gamma});
    this->a.push_back(this->b);

    this->c.push_back(gamma);
    this->c.push_back(0.5*(1+gamma));
    this->c.push_back(1.0);

    break;
  }
  case (SDIRK_3O4): {
    this->n_stages = 3;
    this->a.reserve(this->n_stages);
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);

    double const gamma = 2.0 * std::cos(dealii::numbers::PI / 18.0) / std::sqrt(3.0);

    this->b.push_back(1.0/(6.0*gamma*gamma));
    this->b.push_back(1.0 - 1.0/(3.0*gamma*gamma));
    this->b.push_back(1.0/(6.0*gamma*gamma));

    this->a.push_back(std::vector<double>(1, 0.5*(1.0+gamma)));
    this->a.push_back(std::vector<double>{-0.5*gamma, 0.5*(1.0+gamma)});
    this->a.push_back(std::vector<double>{1.0+gamma,-1.0+2.0*gamma, 0.5*(1+gamma)});

    this->c.push_back(0.5*(1.0+gamma));
    this->c.push_back(0.5);
    this->c.push_back(0.5*(1.0-gamma));

    break;
  }
  case (SDIRK_5O4): {
    this->n_stages = 5;
    this->a.reserve(this->n_stages);
    this->b.reserve(this->n_stages);
    this->c.reserve(this->n_stages);

    this->b.push_back(25./24);
    this->b.push_back(-49./48);
    this->b.push_back(125./16);
    this->b.push_back(-85./12);
    this->b.push_back(1./4);

    this->a.push_back(std::vector<double>(1, 1./4));
    this->a.push_back(std::vector<double>{1./2, 1./4});
    this->a.push_back(std::vector<double>{17./50,-1./25, 1./4});
    this->a.push_back(std::vector<double>{371./1360, -137./2720, 15./544, 1./4});
    this->a.push_back(this->b);

    this->c.push_back(1./4);
    this->c.push_back(3./4);
    this->c.push_back(11./20);
    this->c.push_back(1./2);
    this->c.push_back(1.);

    break;
  }
  default: {
  }
  }
}

template <typename VectorType, typename TimeType>
TimeType ImplicitRungeKutta<VectorType, TimeType>::evolve_one_time_step(
    const std::function<void(const TimeType, const VectorType&, VectorType&)>& f,
    const std::function<void(const TimeType, const TimeType,
                                   const VectorType&, VectorType&)>& id_minus_tau_J_inverse,
    TimeType t, TimeType delta_t, VectorType& y) {
  VectorType              old_y(y);
  std::vector<VectorType> f_stages(this->n_stages, y);
  // Compute the different stages needed.
  compute_stages(f, id_minus_tau_J_inverse, t, delta_t, y, f_stages);

  y = old_y;
  for (unsigned int i = 0; i < this->n_stages; ++i)
    y.sadd(1., delta_t * this->b[i], f_stages[i]);

  return (t + delta_t);
}

template <typename VectorType, typename TimeType>
void ImplicitRungeKutta<VectorType, TimeType>::set_newton_solver_parameters(
    unsigned int max_it_, double tolerance_) {
  max_it    = max_it_;
  tolerance = tolerance_;
}

template <typename VectorType, typename TimeType>
const typename ImplicitRungeKutta<VectorType, TimeType>::Status&
ImplicitRungeKutta<VectorType, TimeType>::get_status() const {
  return status;
}

template <typename VectorType, typename TimeType>
void ImplicitRungeKutta<VectorType, TimeType>::compute_stages(
    const std::function<void(const TimeType, const VectorType&, VectorType&)>& f,
    const std::function<void(const TimeType, const TimeType,
                                   const VectorType&, VectorType&)>& id_minus_tau_J_inverse,
    TimeType t, TimeType delta_t, VectorType& y,
    std::vector<VectorType>& f_stages) {
  VectorType z(y);
  for (unsigned int i = 0; i < this->n_stages; ++i) {
    VectorType old_y(z);
    for (unsigned int j = 0; j < i; ++j)
      old_y.sadd(1., delta_t * this->a[i][j], f_stages[j]);

    // Solve the nonlinear system using Newton's method
    const TimeType new_t       = t + this->c[i] * delta_t;
    const TimeType new_delta_t = this->a[i][i] * delta_t;
    VectorType&  f_stage     = f_stages[i];
    newton_solve(
        [this, &f, new_t, new_delta_t, &old_y, &f_stage](const VectorType& y,
                                                         VectorType& residual) {
          this->compute_residual(f, new_t, new_delta_t, old_y, y, f_stage,
                                 residual);
        },
        [&id_minus_tau_J_inverse, new_t, new_delta_t](const VectorType& yin, VectorType &yout) {
          id_minus_tau_J_inverse(new_t, new_delta_t, yin, yout);
        },
        y);
  }
}

template <typename VectorType, typename TimeType>
void ImplicitRungeKutta<VectorType, TimeType>::newton_solve(
    const std::function<void(const VectorType&, VectorType&)>& get_residual,
    const std::function<void(const VectorType&, VectorType&)>& id_minus_tau_J_inverse,
    VectorType&                                         y) {
  VectorType residual(y);
  get_residual(y, residual);
  unsigned int i                     = 0;
  const double initial_residual_norm = residual.l2_norm();
  double       norm_residual         = initial_residual_norm;
  while (i < max_it) {
    id_minus_tau_J_inverse(residual,residual);
    y.sadd(1.0, -1.0, residual);
    get_residual(y, residual);
    norm_residual = residual.l2_norm();
    if (norm_residual < tolerance)
      break;
    ++i;
  }
  status.n_iterations  = i + 1;
  status.norm_residual = norm_residual;
}

template <typename VectorType, typename TimeType>
void ImplicitRungeKutta<VectorType, TimeType>::compute_residual(
    const std::function<void(const TimeType, const VectorType&, VectorType&)>& f,
    TimeType t, TimeType delta_t, const VectorType& old_y, const VectorType& y,
    VectorType& tendency, VectorType& residual) const {
  // The tendency is stored to save one evaluation of f.
  f(t, y, tendency);
  residual = tendency;
  residual.sadd(delta_t, 1.0, old_y);
  residual.sadd(-1.0, 1., y);
}

/**----------------------------------------------------------------------
 * Exact
 * ----------------------------------------------------------------------
 */
template <typename VectorType, typename TimeType>
TimeType Exact<VectorType, TimeType>::evolve_one_time_step(
    std::vector<std::function<void(const TimeType,      //
                                   const VectorType&, //
                                   VectorType&)>>& /* f */,
    std::vector<std::function<void(const TimeType,      //
                                   const TimeType,      //
                                   const VectorType&, //
                                   VectorType&)>>& exact_eval,
    TimeType t, TimeType delta_t, VectorType& y) {
  // Copy construct input vector
  VectorType old_y(y);
  TimeType     new_t = t + delta_t;
  // Call exact evaluation routine
  exact_eval[0](t, delta_t, old_y, y);
  return new_t;
}

template <typename VectorType,typename TimeType>
const typename Exact<VectorType, TimeType>::Status&
Exact<VectorType,TimeType>::get_status() const {
  return status;
}


/**----------------------------------------------------------------------
 * OperatorSplit
 * ----------------------------------------------------------------------
 */
template <typename BVectorType, typename TimeType>
OperatorSplit<BVectorType, TimeType>::OperatorSplit(
    const std::string                                    in_method_name, //
    const std::vector<OSoperator<BVectorType, TimeType>> in_operators)
    : operators(in_operators), method_name(in_method_name) {
  check_method_name(in_method_name);
}

template <typename BVectorType, typename TimeType>
OperatorSplit<BVectorType, TimeType>::OperatorSplit(
    const std::string                                    in_method_name, //
    const std::vector<OSoperator<BVectorType, TimeType>> in_operators,   //
    const std::vector<OSmask>                            in_mask,        //
    const BVectorType                                    ref)
    : ref_state(ref), operators(in_operators), mask(in_mask),
      nblocks(mask.size()), blockrefs(mask.size()),
      method_name(in_method_name) {

  check_method_name(in_method_name);

  // Allocate space for block refs
  for (size_t i = 0; i < mask.size(); ++i) {
    nblocks[i] = mask[i].size();
    blockrefs[i].reinit(nblocks[i]);
  }

  // Output the masking used for the OS method
  // printf("Blockrefs initialized.\n");
  // printf(" stages: %2lu\n", blockrefs.size());
  // for(size_t i=0; i< mask.size();++i){
  //   printf("  stage %2lu: %2d blocks\n",i, nblocks[i]);
  //   printf("      mask:");
  //   for (int j=0; j< nblocks[i]; ++j) {
  //     printf(" %2d", mask[i][j]);
  //   }
  //   printf("\n");
  // }
  // operators and stages default copy constructed
}

template <typename BVectorType, typename TimeType>
OperatorSplit<BVectorType, TimeType>::OperatorSplit(
    const std::vector<OSoperator<BVectorType, TimeType>> in_operators, //
    const std::vector<OSpair<TimeType>>                  in_stages
)
    : operators(in_operators), stages(in_stages)  {}

template <typename BVectorType, typename TimeType>
OperatorSplit<BVectorType, TimeType>::OperatorSplit(
    const BVectorType                                    in_ref,       //
    const std::vector<OSoperator<BVectorType, TimeType>> in_operators, //
    const std::vector<OSpair<TimeType>>                  in_stages,    //
    const std::vector<OSmask>                            in_mask)
    : ref_state(in_ref), operators(in_operators), stages(in_stages),
      mask(in_mask), nblocks(mask.size()), blockrefs(mask.size()) {
  // Allocate space for block refs
  for (size_t i = 0; i < mask.size(); ++i) {
    nblocks[i] = mask[i].size();
    blockrefs[i].reinit(nblocks[i]);
  }
}

template <typename BVectorType, typename TimeType>
TimeType OperatorSplit<BVectorType, TimeType>::evolve_one_time_step(
    f_vfun_type& /*f*/, jac_vfun_type& /*id_minus_tau_J_inverse*/, TimeType t,
    TimeType delta_t, BVectorType& /*y*/) {
  // Not implemented
  return t + delta_t;
}

template <typename BVectorType, typename TimeType>
TimeType OperatorSplit<BVectorType, TimeType>::evolve_one_time_step(TimeType t, TimeType delta_t,
                                                       BVectorType& y) {
  std::vector<TimeType> op_time(operators.size(), t); // the current time for each operator

  // Loop over stages
  for (const auto& pair : stages) {

    // Get stage info
    auto op    = pair.op_num;
    auto alpha = pair.alpha;

    // Operator info for this stage
    auto          method = operators[op].method;
    f_vfun_type   function{operators[op].function};
    jac_vfun_type id_minus_tau_J_inverse{operators[op].id_minus_tau_J_inverse};

    // Update blockref pointers for this stage's state
    auto m = mask[op];
    for(int j=0;j<nblocks[op];++j) {
      std::swap(blockrefs[op].block(j), y.block(m[j]));
    }

    //--------------------------------------------
    //   DEBUG: checking masking setup
    //   --------------
    // std::ofstream outf_b;
    // outf_b.open("blockrefs.out");
    // blockrefs[op].print(outf_b);
    // std::ofstream outf_y;
    // outf_y.open("y.out");
    // y.print(outf_y);
    // ---------------------------

    // Evolve this operator with the masked sub-blocks
    op_time[op] = method->evolve_one_time_step(function, //
					       id_minus_tau_J_inverse, //
                                               op_time[op], //
					       alpha * delta_t, //
					       blockrefs[op]);

    // Swap blocks back
    for(int j=0;j<nblocks[op];++j) {
      std::swap(y.block(m[j]),blockrefs[op].block(j));
    }

  }

  // if(t>0.1)
  //   exit(0);

  return (t + delta_t);
}

template <typename BVectorType, typename TimeType>
const typename OperatorSplit<BVectorType, TimeType>::Status&
OperatorSplit<BVectorType, TimeType>::get_status() const {
  return status;
}


/**----------------------------------------------------------------------
 * OperatorSplitSingle
 * ----------------------------------------------------------------------
 */
template <typename VectorType, typename TimeType>
OperatorSplitSingle<VectorType, TimeType>::OperatorSplitSingle(
    const VectorType                                    ref,          //
    const std::vector<OSoperator<VectorType, TimeType>> in_operators, //
    const std::vector<OSpair<TimeType>>                 in_stages)
    : ref_state(ref), operators(in_operators), stages(in_stages) {
  // operators and stages default copy constructed
}

template <typename VectorType, typename TimeType>
TimeType OperatorSplitSingle<VectorType, TimeType>::evolve_one_time_step(
    f_vfun_type& /*f*/,                        //
    jac_vfun_type& /*id_minus_tau_J_inverse*/, //
    TimeType t,                                //
    TimeType delta_t,                          //
    VectorType& /*y*/) {
  // Not implemented
  return t + delta_t;
}

template <typename VectorType, typename TimeType>
TimeType OperatorSplitSingle<VectorType, TimeType>::evolve_one_time_step(
    TimeType    t,       //
    TimeType    delta_t, //
    VectorType& y) {

  // the current time for each operator
  std::vector<TimeType> op_time(operators.size(), t);

  // Loop over stages
  for (const auto& pair : stages) {

    // Get stage info
    auto op    = pair.op_num;
    auto alpha = pair.alpha;

    // Operator info for this stage
    auto          method = operators[op].method;
    f_vfun_type   function{operators[op].function};
    jac_vfun_type id_minus_tau_J_inverse{operators[op].id_minus_tau_J_inverse};

    // Evolve this operator with the masked sub-blocks
    op_time[op] = method->evolve_one_time_step(function,               //
                                               id_minus_tau_J_inverse, //
                                               op_time[op],            //
                                               alpha * delta_t,        //
                                               y);
  }

  return (t + delta_t);
}

template <typename VectorType, typename TimeType>
const typename OperatorSplitSingle<VectorType, TimeType>::Status&
OperatorSplitSingle<VectorType, TimeType>::get_status() const {
  return status;
}

template <typename VectorType, typename TimeType>
void OperatorSplit<VectorType, TimeType>::check_method_name(std::string name){

  try {
    os_method.at(name);      // throws if key not found
  }
  catch (const std::out_of_range& oor1) {
    try {
      os_complex.at(name);
    }
    catch (const std::out_of_range& oor2) {

      std::cerr << "OS method (" << name << ") not found in available methods.\n";
      std::cerr << " Available methods (standard):\n";
      for (auto  it : os_method)
	std:: cerr << "  " << it.first << "\n";
      std::cerr << " Available methods (complex):\n";
      for (auto  it : os_complex)
	std:: cerr << "  " << it.first << "\n";

      throw oor2;
    }
  }
}

std::string RK_method_enum_to_string(runge_kutta_method method) {
  std::string method_name;
  switch (method) {
  case FORWARD_EULER: {
    method_name = "FORWARD_EULER";
    break;
  }
  case EXPLICIT_MIDPOINT: {
    method_name = "EXPLICIT_MIDPOINT";
    break;
  }
  case HEUN2: {
    method_name = "HEUN2";
    break;
  }
  case RK_THIRD_ORDER: {
    method_name = "RK_THIRD_ORDER";
    break;
  }
  case RK_CLASSIC_FOURTH_ORDER: {
    method_name = "RK_CLASSIC_FOURTH_ORDER";
    break;
  }
  case BACKWARD_EULER: {
    method_name = "BACKWARD_EULER";
    break;
  }
  case IMPLICIT_MIDPOINT: {
    method_name = "IMPLICIT_MIDPOINT";
    break;
  }
  case CRANK_NICOLSON: {
    method_name = "CRANK_NICOLSON";
    break;
  }
  case SDIRK_TWO_STAGES: {
    method_name = "SDIRK_TWO_STAGES";
    break;
  }
  case SDIRK_THREE_STAGES: {
    method_name = "SDIRK_THREE_STAGES";
    break;
  }
  case SDIRK_3O4: {
    method_name = "SDIRK_3O4";
    break;
  }
  case SDIRK_5O4: {
    method_name = "SDIRK_5O4";
    break;
  }
  default: {
    break;
  }
  }

  return method_name;
}

runge_kutta_method RK_string_to_enum(std::string method_name) {
  runge_kutta_method method;
  if (method_name.compare("FORWARD_EULER") == 0) {
    method = FORWARD_EULER;
  }
  if (method_name.compare("EXPLICIT_MIDPOINT") == 0) {
    method = EXPLICIT_MIDPOINT;
  }
  if (method_name.compare("HEUN2") == 0) {
    method = HEUN2;
  }
  if (method_name.compare("RK_THIRD_ORDER") == 0) {
    method = RK_THIRD_ORDER;
  }
  if (method_name.compare("RK_CLASSIC_FOURTH_ORDER") == 0) {
    method = RK_CLASSIC_FOURTH_ORDER;
  }
  if (method_name.compare("BACKWARD_EULER") == 0) {
    method = BACKWARD_EULER;
  }
  if (method_name.compare("IMPLICIT_MIDPOINT") == 0) {
    method = IMPLICIT_MIDPOINT;
  }
  if (method_name.compare("CRANK_NICOLSON") == 0) {
    method = CRANK_NICOLSON;
  }
  if (method_name.compare("SDIRK_TWO_STAGES") == 0) {
    method = SDIRK_TWO_STAGES;
  }
  if (method_name.compare("SDIRK_THREE_STAGES") == 0) {
    method = SDIRK_THREE_STAGES;
  }
  if (method_name.compare("SDIRK_3O4") == 0) {
    method = SDIRK_3O4;
  }
  if (method_name.compare("SDIRK_5O4") == 0) {
    method = SDIRK_5O4;
  }

  return method;
}




/**
 * Compile-time template declarations
 */

template class ExplicitRungeKutta<dealii::Vector<double>>;
template class ImplicitRungeKutta<dealii::Vector<double>>;
template class Exact<dealii::Vector<double>>;

// Complex-valued, with real-valud time
template class ExplicitRungeKutta<dealii::Vector<std::complex<double>>>;
template class ImplicitRungeKutta<dealii::Vector<std::complex<double>>>;
template class Exact<dealii::Vector<std::complex<double>>>;

// Complex-valued, with complex-valued time
template class ExplicitRungeKutta<dealii::Vector<std::complex<double>>,std::complex<double>>;
template class ImplicitRungeKutta<dealii::Vector<std::complex<double>>,std::complex<double>>;
template class Exact<dealii::Vector<std::complex<double>>,std::complex<double>>;
// Block vector, Complex-valued, with complex-valued time
template class ExplicitRungeKutta<dealii::BlockVector<std::complex<double>>,std::complex<double>>;
template class ImplicitRungeKutta<dealii::BlockVector<std::complex<double>>,std::complex<double>>;
template class Exact<dealii::BlockVector<std::complex<double>>,std::complex<double>>;


template class ExplicitRungeKutta<dealii::BlockVector<double>>;
template class ImplicitRungeKutta<dealii::BlockVector<double>>;

template class ExplicitRungeKutta<dealii::PETScWrappers::MPI::Vector>;
template class ImplicitRungeKutta<dealii::PETScWrappers::MPI::Vector>;

template class ExplicitRungeKutta<dealii::PETScWrappers::MPI::BlockVector>;
template class ImplicitRungeKutta<dealii::PETScWrappers::MPI::BlockVector>;
// template class ExplicitRungeKutta<dealii::PETScWrappers::MPI::Vector>;
// template class ImplicitRungeKutta<dealii::PETScWrappers::MPI::Vector>;

// OS types
template class OperatorSplit<dealii::BlockVector<double>>;
template class OperatorSplit<dealii::BlockVector<std::complex<double>>,std::complex<double>>;
template class OperatorSplit<dealii::PETScWrappers::MPI::BlockVector>;
// template class OperatorSplit<dealii::PETScWrappers::MPI::Vector>;

template class OperatorSplitSingle<dealii::Vector<std::complex<double>>>;
template class OperatorSplitSingle<dealii::Vector<std::complex<double>>,std::complex<double>>;

}
