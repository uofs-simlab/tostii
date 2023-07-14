#include <tostii/time_stepping/operator_split.inl>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_block_vector.h>

namespace tostii::TimeStepping
{
    const std::array<double, 2>
    os_method<double>::yoshida_omega = { {
        -1.702414383919315268,
         1.351207191959657634
    } };

    const std::array<double, 3>
    os_method<double>::a_pp3a3 = { {
         0.461601939364879971,
        -0.0678710530507800810, 
        -0.0958868852260720250
    } };
    const std::array<double, 3>
    os_method<double>::b_pp3a3 = { {
        -0.266589223588183997,
         0.0924576733143338350,
         0.674131550273850162
    } };
    const std::array<double, 3>
    os_method<double>::c_pp3a3 = { {
        -0.360420727960349671,
         0.579154058410941403,
         0.483422668461380403
    } };

    const std::array<std::pair<std::string, std::vector<OSPair<double>>>, os_method<double>::INVALID>
    os_method<double>::info = { {
        { "Godunov", {
            { 0, 1.0 },
            { 1, 1.0 }
        } },
        { "Strang", {
            { 0, 0.5 },
            { 1, 1.0 },
            { 0, 0.5 }
        } },
        { "Ruth", {
            { 0,  7.0 / 24.0 },
            { 1,  2.0 /  3.0 },
            { 0,  3.0 /  4.0 },
            { 1, -2.0 /  3.0 },
            { 0, -1.0 / 24.0 },
            { 1,  1.0        }
        } },
        { "Yoshida", {
            { 0,                     yoshida_omega[1]  / 2.0 },
            { 1,                     yoshida_omega[1]        },
            { 0, (yoshida_omega[0] + yoshida_omega[1]) / 2.0 },
            { 1,  yoshida_omega[0]                           },
            { 0, (yoshida_omega[0] + yoshida_omega[1]) / 2.0 },
            { 1,                     yoshida_omega[1]        },
            { 0,                     yoshida_omega[1]  / 2.0 }
        } },
        { "Godunov3", {
            { 0, 1.0 },
            { 1, 1.0 },
            { 2, 1.0 }
        } },
        { "Strang3", {
            { 0, 0.5 },
            { 1, 0.5 },
            { 2, 1.0 },
            { 1, 0.5 },
            { 0, 0.5 }
        } },
        { "PP_3_A_3", {
            { 0, a_pp3a3[0] },
            { 1, b_pp3a3[0] },
            { 2, c_pp3a3[0] },
            { 0, a_pp3a3[1] },
            { 1, b_pp3a3[1] },
            { 2, c_pp3a3[1] },
            { 0, a_pp3a3[2] },
            { 1, b_pp3a3[2] },
            { 2, c_pp3a3[2] },
            { 0, c_pp3a3[2] },
            { 1, b_pp3a3[2] },
            { 2, a_pp3a3[2] },
            { 0, c_pp3a3[1] },
            { 1, b_pp3a3[1] },
            { 2, a_pp3a3[1] },
            { 0, c_pp3a3[0] },
            { 1, b_pp3a3[0] },
            { 2, a_pp3a3[0] }
        } }
    } };

    const std::unordered_map<std::string, os_method_t<double>>
    os_method<double>::values = { {
        { "Godunov",  type::GODUNOV  },
        { "Strang",   type::STRANG   },
        { "Ruth",     type::RUTH     },
        { "Yoshida",  type::YOSHIDA  },
        { "Godunov3", type::GODUNOV3 },
        { "Strang3",  type::STRANG3  },
        { "PP_3_A_3", type::PP_3_A_3 }
    } };

    typename os_method<double>::type os_method<double>::from_string(const std::string& name)
    {
        return values.at(name);
    }

    const std::string& os_method<double>::to_string(const type method)
    {
        return info[method].first;
    }

    const std::vector<OSPair<double>>& os_method<double>::to_os_pairs(const type method)
    {
        return info[method].second;
    }

    using std::complex_literals::operator""i;

    const std::array<std::complex<double>, 3>
    os_method<std::complex<double>>::a_pp3a3c = { {
        0.0442100822731214750 - 0.0713885293035937610i,
        0.157419072651724312  - 0.1552628290245811054i,
        0.260637333463417766  + 0.07744172526769638060i
    } };
    const std::array<std::complex<double>, 3>
    os_method<std::complex<double>>::b_pp3a3c = { {
        0.0973753110633760580 - 0.112390152630243038i,
        0.179226865237094561  - 0.0934263750859694960i,
        0.223397823699529381  + 0.205816527716212534i
    } };
    const std::array<std::complex<double>, 3>
    os_method<std::complex<double>>::c_pp3a3c { {
        0.125415464915697242 - 0.281916718734615225i,
        0.353043498499040389 + 0.0768951336684972038i,
        0.059274548196998816 + 0.354231218126596507i
    } };

    const std::array<
        std::pair<std::string, std::vector<OSPair<std::complex<double>>>>,
        os_method<std::complex<double>>::INVALID>
    os_method<std::complex<double>>::info = { { 
        { "Milne_2_2_c_i", {
            { 0, 12.0 / 37.0 - 2.0i / 37.0 },
            { 1, 25.0 / 34.0 - 1.0i / 17.0 },
            { 0, 25.0 / 37.0 + 2.0i / 37.0 },
            { 1,  9.0 / 34.0 + 1.0i / 17.0 }
        } },
        { "Milne_2_2_c_i_asc", {
            { 0, 0.8 - 0.4i },
            { 1, 0.5 + 1.0i },
            { 0, 0.2 + 0.4i },
            { 1, 0.5 - 1.0i }
        } },
        { "Milne_2_2_c_ii", {
            { 0,  4.0 / 13.0 - 1.0i / 26.0 },
            { 1, 18.0 / 25.0 - 1.0i / 25.0 },
            { 0,  9.0 / 13.0 + 1.0i / 26.0 },
            { 1,  7.0 / 25.0 + 1.0i / 25.0 }
        } },
        { "A_3_3_c", {
            { 1, 0.25 + 0.144337567297406441i },
            { 0,  0.5 + 0.288675134594812882i },
            { 1,  0.5                         },
            { 0,  0.5 - 0.288675134594812882i },
            { 1, 0.25 - 0.144337567297406441i }
        } },
        { "AKT_2_2_c", {
            { 0, 0.5 + 0.5i },
            { 1, 0.5 + 0.5i },
            { 2, 0.5 + 0.5i },
            { 0, 0.5 - 0.5i },
            { 1, 0.5 - 0.5i },
            { 2, 0.5 - 0.5i }
        } },
        { "PP_3_A_3_c", {
            { 0, a_pp3a3c[0] },
            { 1, b_pp3a3c[0] },
            { 2, c_pp3a3c[0] },
            { 0, a_pp3a3c[1] },
            { 1, b_pp3a3c[1] },
            { 2, c_pp3a3c[1] },
            { 0, a_pp3a3c[2] },
            { 1, b_pp3a3c[2] },
            { 2, c_pp3a3c[2] },
            { 0, c_pp3a3c[2] },
            { 1, b_pp3a3c[2] },
            { 2, a_pp3a3c[2] },
            { 0, c_pp3a3c[1] },
            { 1, b_pp3a3c[1] },
            { 2, a_pp3a3c[1] },
            { 0, c_pp3a3c[0] },
            { 1, b_pp3a3c[0] },
            { 2, a_pp3a3c[0] }
        } }
    } };

    const std::unordered_map<std::string, os_method_t<std::complex<double>>>
    os_method<std::complex<double>>::values = { { 
        { "Milne_2_2_c_i",     type::MILNE_2_2_C_I },
        { "Milne_2_2_c_i_asc", type::MILNE_2_2_C_I_ASC },
        { "Milne_2_2_c_ii",    type::MILNE_2_2_C_II },
        { "A_3_3_c",           type::A_3_3_C },
        { "AKT_2_2_c",         type::AKT_2_2_C },
        { "PP_3_A_3_c",        type::PP_3_A_3_C }
    } };

    typename os_method<std::complex<double>>::type os_method<std::complex<double>>::from_string(const std::string& name)
    {
        return values.at(name);
    }

    const std::string& os_method<std::complex<double>>::to_string(const type method)
    {
        return info[method].first;
    }

    const std::vector<OSPair<std::complex<double>>>& os_method<std::complex<double>>::to_os_pairs(const type method)
    {
        return info[method].second;
    }

    template class OperatorSplit<dealii::BlockVector<double>>;
    template class OperatorSplit<dealii::BlockVector<std::complex<double>>>;
    template class OperatorSplit<dealii::BlockVector<std::complex<double>>, std::complex<double>>;

    template class OperatorSplit<dealii::PETScWrappers::MPI::BlockVector>;
}
