#!/bin/bash

# Sample run script template

MAKE_PROCESSES=4
MPI_PROCESSES=4
PROBLEM="bidomain"
TARGET="bidomain_godunov"
EXECUTABLE="out/${PROBLEM}/bin/${TARGET}"

if [ ! -d out ]
then
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -S . -B out"
else
    CMAKE_ARGS="out"
fi

cmake ${CMAKE_ARGS} 2>&1
CMAKE_STATUS=$?

if [ $CMAKE_STATUS -ne 0 ]
then
    printf "\nCMake exited with code %d.\n" ${CMAKE_STATUS}
    exit 1
fi

make -j${MAKE_PROCESSES} -sC out ${TARGET}
MAKE_STATUS=$?

if [ $MAKE_STATUS -ne 0 ]
then
    printf "\nmake exited with code %d.\n" ${MAKE_STATUS}
    exit 1
fi

if [ -n ${MPI_PROCESSES} -a ${MPI_PROCESSES} -gt 1 ]
then
    mpiexec -n ${MPI_PROCESSES} ${EXECUTABLE} $@
else
    ${EXECUTABLE} $@
fi
