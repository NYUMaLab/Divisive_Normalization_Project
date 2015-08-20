#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00
#PBS -l mem=2GB
#PBS -N python
#PBS -M david.halpern@nyu.edu
#PBS -j oe
 
module purge
module load python

RUNDIR=$SCRATCH/my_project/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR
 
DATADIR=$SCRATCH/my_project/data
cd $RUNDIR
stata -b do $DATADIR/data_0706.do