#!/bin/bash

# Project
#PBS -P BGH

# 128 CPUs
#PBS -l select=16:ncpus=8:mpiprocs=8:mem=60GB

#PBS -l walltime=00:20:00
#PBS -M tristan.salles@sydney.edu.au
#PBS -m abe
#PBS -q alloc-dm

# set up environment
module load gcc/4.9.3 python/3.6.5 petsc-gcc-mpich/3.11.1

cd $PBS_O_WORKDIR
cd dataAnalysis

# Launching the job!
mpirun -np 128 python getCatchmentInfo.py -i inputSedFlow6.csv -o flowsed_model6 -v
