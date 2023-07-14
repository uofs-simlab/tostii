#include "bidomain_godunov.inl"

int main(int argc, char* argv[])
try
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    if (argc != 2)
    {
        using std::string_literals::operator""s;
        throw std::invalid_argument("Usage: "s + argv[0] + " param_file");
    }

    Bidomain::Parameters::AllParameters param(argv[1]);
    if (param.dim < 2 || param.dim > 3)
    {
        throw std::invalid_argument("Bad dim value");
    }

    switch (param.dim)
    {
    case 2:
        {
            Bidomain::BidomainProblem<2> problem(param);
            problem.run();
        }
        break;
    case 3:
        {
            Bidomain::BidomainProblem<3> problem(param);
            problem.run();
        }
        break;
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
