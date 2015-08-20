#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=5:00:00
#PBS -l mem=2GB
#PBS -N jobname
#PBS -M bob.smith@nyu.edu
#PBS -j oe
 
module purge
 
SRCDIR=$HOME/my_project/code
RUNDIR=$SCRATCH/my_project/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR
 
cd $PBS_O_WORKDIR
cp my_input_params.inp $RUNDIR
 
cd $RUNDIR
module load fftw/intel/3.3
$SRCDIR/my_exec.exe < my_input_params.inp