#pragma once

#include "utils.h"

namespace NSE::DataPostprocessors
{
    using std::string_literals::operator""s;

	template<int dim>
	ComplexRealPart<dim>::ComplexRealPart(
        const char* name)
		: DataPostprocessorScalar<dim>(name + "_Re"s, update_values)
	{ }

	template<int dim>
	void ComplexRealPart<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& inputs,
		std::vector<Vector<double>>& computed_quantities) const
	{
		AssertDimension(computed_quantities.size(), inputs.solution_values.size());

		for (unsigned int q = 0; q < computed_quantities.size(); ++q)
		{
			AssertDimension(computed_quantities[q].size(), 1);
			AssertDimension(inputs.solution_values[q].size(), 2);

			computed_quantities[q](0) = inputs.solution_values[q](0);
		}
	}

	template<int dim>
	ComplexImagPart<dim>::ComplexImagPart(const char* name)
		: DataPostprocessorScalar<dim>(name + "_Im"s, update_values)
	{ }

	template<int dim>
	void ComplexImagPart<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& inputs,
		std::vector<Vector<double>>& computed_quantities) const
	{
		AssertDimension(computed_quantities.size(), inputs.solution_values.size());

		for (unsigned int q = 0; q < computed_quantities.size(); ++q)
		{
			AssertDimension(computed_quantities[q].size(), 1);
			AssertDimension(inputs.solution_values[q].size(), 2);

			computed_quantities[q](0) = inputs.solution_values[q](1);
		}
	}

	template<int dim>
	ComplexAmplitude<dim>::ComplexAmplitude(
        const char* name)
		: DataPostprocessorScalar<dim>(name + "_Amplitude"s, update_values)
	{

	}

	template<int dim>
	void ComplexAmplitude<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& inputs,
		std::vector<Vector<double>>& computed_quantities) const
	{
		AssertDimension(computed_quantities.size(), inputs.solution_values.size());

		for (unsigned int q = 0; q < computed_quantities.size(); ++q)
		{
			AssertDimension(computed_quantities[q].size(), 1);
			AssertDimension(inputs.solution_values[q].size(), 2);

			const std::complex<double> psi(inputs.solution_values[q](0), inputs.solution_values[q](1));
			computed_quantities[q](0) = std::norm(psi);
		}
	}

	template<int dim>
	ComplexPhase<dim>::ComplexPhase(
        const char* name)
		: DataPostprocessorScalar<dim>(name + "_Phase"s, update_values)
	{ }

	template<int dim>
	void ComplexPhase<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& inputs,
		std::vector<Vector<double>>& computed_quantities) const
	{
		AssertDimension(computed_quantities.size(), inputs.solution_values.size());

		double max_phase = -numbers::PI;
		for (unsigned int q = 0; q < computed_quantities.size(); ++q)
		{
			AssertDimension(computed_quantities[q].size(), 1);
			AssertDimension(inputs.solution_values[q].size(), 2);

			const std::complex<double> value(inputs.solution_values[q](0), inputs.solution_values[q](1));
			max_phase = std::max(max_phase, std::arg(value));
		}

		for (auto& output : computed_quantities)
		{
			output(0) = max_phase;
		}
	}
}
