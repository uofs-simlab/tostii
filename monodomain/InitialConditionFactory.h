#ifndef TOSTII_MONODOMAIN_INITIAL_CONDITION_FACTORY_H
#define TOSTII_MONODOMAIN_INITIAL_CONDITION_FACTORY_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace InitialConditions
{
  using namespace dealii;

  struct Parameters
  {
    std::string type = "disk";
    double      center_x = 0.0;
    double      center_y = 0.0;
    double      amplitude_v = 0.2;
    double      amplitude_w = 0.0;
    double      radius = 2.0;
    double      smooth_width = 0.5;
    double      gaussian_width = 1.0;
    double      steepness_x = 2.0;
    double      steepness_y = 2.0;

    static void declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Initial condition");
      {
        prm.declare_entry("Type",
                          "disk",
                          Patterns::Selection(
                            "zero|constant|disk|gaussian|broken_wave|spiral"),
                          "Initial-condition type.");
        prm.declare_entry("Center x",
                          "0.0",
                          Patterns::Double(),
                          "Initial-condition center x-coordinate.");
        prm.declare_entry("Center y",
                          "0.0",
                          Patterns::Double(),
                          "Initial-condition center y-coordinate.");
        prm.declare_entry("Amplitude v",
                          "0.2",
                          Patterns::Double(),
                          "Initial-condition amplitude for v.");
        prm.declare_entry("Amplitude w",
                          "0.0",
                          Patterns::Double(),
                          "Initial-condition amplitude for w.");
        prm.declare_entry("Radius",
                          "2.0",
                          Patterns::Double(0.0),
                          "Disk radius.");
        prm.declare_entry("Smooth width",
                          "0.5",
                          Patterns::Double(0.0),
                          "Disk edge smoothing width.");
        prm.declare_entry("Gaussian width",
                          "1.0",
                          Patterns::Double(0.0),
                          "Gaussian width parameter.");
        prm.declare_entry("Steepness x",
                          "2.0",
                          Patterns::Double(0.0),
                          "Steepness of the x-directed broken-wave interface.");
        prm.declare_entry("Steepness y",
                          "2.0",
                          Patterns::Double(0.0),
                          "Steepness of the y-directed refractory interface.");
      }
      prm.leave_subsection();
    }

    void parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Initial condition");
      {
        type           = prm.get("Type");
        center_x       = prm.get_double("Center x");
        center_y       = prm.get_double("Center y");
        amplitude_v    = prm.get_double("Amplitude v");
        amplitude_w    = prm.get_double("Amplitude w");
        radius         = prm.get_double("Radius");
        smooth_width   = prm.get_double("Smooth width");
        gaussian_width = prm.get_double("Gaussian width");
        steepness_x    = prm.get_double("Steepness x");
        steepness_y    = prm.get_double("Steepness y");
      }
      prm.leave_subsection();
    }
  };

  template <int dim>
  class ZeroInitialCondition : public Function<dim>
  {
  public:
    ZeroInitialCondition()
      : Function<dim>(2)
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/) const override
    {
      return 0.0;
    }
  };

  template <int dim>
  class ConstantInitialCondition : public Function<dim>
  {
  public:
    explicit ConstantInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component) const override
    {
      if (component == 0)
        return prm.amplitude_v;
      if (component == 1)
        return prm.amplitude_w;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  template <int dim>
  class DiskInitialCondition : public Function<dim>
  {
  public:
    explicit DiskInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double dx = p[0] - prm.center_x;
      const double dy = p[1] - prm.center_y;
      const double r  = std::sqrt(dx * dx + dy * dy);

      const double sigma = std::max(prm.smooth_width, 1.0e-12);
      const double profile =
        0.5 * (1.0 - std::tanh((r - prm.radius) / sigma));

      if (component == 0)
        return prm.amplitude_v * profile;
      if (component == 1)
        return prm.amplitude_w * profile;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  template <int dim>
  class GaussianInitialCondition : public Function<dim>
  {
  public:
    explicit GaussianInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double dx = p[0] - prm.center_x;
      const double dy = p[1] - prm.center_y;
      const double width = std::max(prm.gaussian_width, 1.0e-12);
      const double profile =
        std::exp(-(dx * dx + dy * dy) / (width * width));

      if (component == 0)
        return prm.amplitude_v * profile;
      if (component == 1)
        return prm.amplitude_w * profile;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  template <int dim>
  class BrokenWaveInitialCondition : public Function<dim>
  {
  public:
    explicit BrokenWaveInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double x = p[0] - prm.center_x;
      const double y = p[1] - prm.center_y;

      const double kx = std::max(prm.steepness_x, 1.0e-12);
      const double ky = std::max(prm.steepness_y, 1.0e-12);

      const double v_profile = 0.5 * (1.0 - std::tanh(kx * x));
      const double w_profile = 0.5 * (1.0 - std::tanh(ky * y));

      if (component == 0)
        return prm.amplitude_v * v_profile;
      if (component == 1)
        return prm.amplitude_w * w_profile;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  template <int dim>
  std::unique_ptr<Function<dim>>
  make_initial_condition(const Parameters &prm)
  {
    if (prm.type == "zero")
      return std::make_unique<ZeroInitialCondition<dim>>();
    if (prm.type == "constant")
      return std::make_unique<ConstantInitialCondition<dim>>(prm);
    if (prm.type == "disk")
      return std::make_unique<DiskInitialCondition<dim>>(prm);
    if (prm.type == "gaussian")
      return std::make_unique<GaussianInitialCondition<dim>>(prm);
    if (prm.type == "broken_wave" || prm.type == "spiral")
      return std::make_unique<BrokenWaveInitialCondition<dim>>(prm);

    throw std::runtime_error("Unsupported initial-condition type: " + prm.type);
  }
} // namespace InitialConditions

#endif
