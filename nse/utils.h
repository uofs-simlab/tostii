#pragma once

#include <deal.II/numerics/data_out.h>

#include "parameters.h"

namespace NSE::DataPostprocessors
{
    template<int dim>
	class ComplexRealPart
		: public DataPostprocessorScalar<dim>
	{
	public:
		ComplexRealPart(
            const char* name);

		void evaluate_vector_field(
            const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const override;
	};

    template<int dim>
	class ComplexImagPart
		: public DataPostprocessorScalar<dim>
	{
	public:
		ComplexImagPart(
            const char* name);

		void evaluate_vector_field(
            const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const override;
	};

    template<int dim>
	class ComplexAmplitude
		: public DataPostprocessorScalar<dim>
	{
	public:
		ComplexAmplitude(
            const char* name);

		void evaluate_vector_field(
            const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const override;
	};

    template<int dim>
	class ComplexPhase
		: public DataPostprocessorScalar<dim>
	{
	public:
		ComplexPhase(const char* name);

		void evaluate_vector_field(
            const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const override;
	};
}
