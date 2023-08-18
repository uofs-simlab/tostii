[![DOI](https://zenodo.org/badge/589803608.svg)](https://zenodo.org/badge/latestdoi/589803608)

# tost.II

Source code for the Temporal Operator-Splitting Template library
(tost.II).


## Build

Requires a working installation of deal.II >= v9.4. This can be
built as follows:

1. Clone the "Compile and Install" project from git:
```sh
# avoid unpacking files on the network filesystem
cd /some/local/dir
git clone https://github.com/dealii/candi.git
cd candi
```

2. Make sure that you have the necessary packages installed:
```sh
cd deal.II-toolchain/platforms/supported
# read the documentation for your platform, e.g.:
vim ubuntu.platform
# follow the instructions, if applicable, and return to the main directory
cd ../../..
```

3. Edit candi's configuration options.
For example, uncomment the line that reads `#PACKAGES="${PACKAGES} once:sundials"` (needed for this project),
and comment out the line that reads `PACKAGES="${PACKAGES} once:slepc"` (because it doesn't work for candi).
```sh
vim candi.cfg
```

4. Run the installer. This will take a long time, so use `screen` if on a remote machine.
```sh
# set number of make processes
BUILD_PROCESSES=4
# set installation prefix - again, avoid network filesystem
CANDI_INSTALL_PREFIX=/home/brl423/dealii
# run with -y to skip confirmations
# run with PACKAGES_OFF=slepc because it doesn't work
PACKAGES_OFF=slepc ./candi.sh --prefix=${CANDI_INSTALL_PREFIX} -j${BUILD_PROCESSES} -y
```

5. Whenever you need to build something with deal.II, first use:
```sh
source ${CANDI_INSTALL_PREFIX}/configuration/enable.sh
```

If the above fails, it is likely due to system packages that are not
installed on your system. The installer will suggest a list of
packages for you to install from your operating system's repositories.

With an appropriate version of deal.II (>= v9.4.0) installed, you can build this
project via cmake:
```sh
# first-time setup:
cmake -DCMAKE_BUILD_TYPE=Release -S . -B out
# on future runs:
cmake out
# build project using make
make -j${BUILD_PROCESSES} -sC out
```

If CMake produces an error during the first command, it is possible that your version of CMake is too old.
You can update the system's CMake if you have root access,
or install a more recent version on your account if you don't.
For example, it seems that CMake version 3.26 works, but 3.12 doesn't
(which is annoying if your package manager only provides a non-working version by default).

## Execution

There are a number of example/test programs using deal.II/tostii included in this repository, in the directories:
 - `neutron` (See deal.II step-52)
 - `nse` (See deal.II step-58)
 - `monodomain` (Cardiac simulation - Monodomain cell model)
 - `bidomain` (Cardiac simulation - Bidomain cell model)
A program defined in `tostii/dir/prog.cc` has its executable saved to `tostii/out/dir/bin/prog`.
It is recommended that you write a run script, like `run.sh`, to automate building and running programs if needed.
Some directory contain a `doc/` directory which contains LaTeX documentation on the directories contents;
it is highly recommended to read the documentation before the code.
