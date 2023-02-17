[![DOI](https://zenodo.org/badge/589803608.svg)](https://zenodo.org/badge/latestdoi/589803608)

# tost.II

Source code for the Temporal Operator-Splitting Template library
(tost.II).


## Build

Requires a working installation of deal.II >= v9.4. This can be
built (tested on Ubuntu, Fedora) via the command
```
	# Obtain a modified candi builder for deal.II, build it
    git submodule update --init
	cd candi
	./candi.sh -j N -y   # use N build threads, confirm default options
```

If the above fails, it is likely due to system packages that are not
installed on your system. The installer will suggest a list of
packages for you to install from your operating system's repositories.

*Note* that the candi build process for deal.II can take awhile,
approximately 30 minutes on a high end workstation.

With an appropriate version of deal.II (>= v9.4.0) installed, you can build this
project via cmake:
```
	# Building in source is fine
	cmake -DDEAL_II_DIR=$HOME/dealii-candi/deal.II-v9.4.1 -DCMAKE_BUILD_TYPE=Release .
	make -j
```

## Execution

5 executables should be built corresponding to the source files (*.cc)
associated with them:
- neutron
- nse
- nse_three
- monodomain
- monodomain_direct
