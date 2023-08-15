#include "monolithic_hs.inl"

#include <deal.II/base/utilities.h>

int main(int argc, char* argv[])
try
{
	dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	if (argc != 2)
	{
		using std::string_literals::operator""s;
		throw std::invalid_argument("Usage: "s + argv[0] + " parameter_file");
	}

	NSE::Parameters::AllParameters param(argv[1]);

	NSE::NonlinearSchroedingerEquation<2> problem(param);
	problem.run();

	return 0;
}
catch (std::exception& exc)
{
	std::cerr << "----------------------------------------\n"
	             "Uncaught exception in main:\n"
	          << exc.what() << "\nAborting!\n"
	             "----------------------------------------\n";
	return 1;
}
catch (...)
{
	std::cerr << "----------------------------------------\n"
	             "Uncaught error in main\nAborting!\n"
	             "----------------------------------------\n";
	return 1;
}
