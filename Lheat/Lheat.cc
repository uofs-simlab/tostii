#include <deal.II/base/mpi.h>

#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
try
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    if (argc != 2)
    {
        using std::string_literals::operator""s;
        throw std::invalid_argument("Usage: "s + argv[0] + " param_file");
    }

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
