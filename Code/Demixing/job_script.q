#!/bin/bash                                                                                                                                                                                                                                                                                                                       
#PBS -l nodes=1:ppn=4                                                                                                                                              
#PBS -l walltime=5:00:00                                                                                                                                                                   
#PBS -l mem=5GB                                                                                                                                                                                                                                                                                                                   
#PBS -N python                                                                                                                                                                                                                                                                                                                
#PBS -M david.halpern@nyu.edu                                                                                                                                                                                                                                                                                                   
#PBS -j oe                                                                                                                                                                                                                                                                                                                     
#PBS -lfeature=ivybridge
#PBS -t 1-90                                                                                                                                                      
index=${PBS_ARRAYID}
job=${PBS_JOBID}                                                                                                                                                                                                                                                                                                

module purge
module load numpy/intel/1.8.1
module load cython/intel/0.22
module load theano/20150721

SRCDIR=$HOME/Divisive_Normalization_Project/demixing_project/Code/Demixing/
RUNDIR=$SCRATCH/my_project/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $RUNDIR
rm -rf ~/.theano

THEANO_FLAGS="base_compiledir=$RUNDIR/compiledir/$index/ " python $SRCDIR/random_networks.py $index

echo "Done"