#!/bin/bash                                                                                                                                                       
#PBS -l nodes=1:ppn=4                                                                                                                                             
#PBS -l walltime=2:00:00                                                                                                                                          
#PBS -l mem=2GB                                                                                                                                                   
#PBS -N python                                                                                                                                                    
#PBS -M david.halpern@nyu.edu                                                                                                                                     
#PBS -j oe                                                                                                                                                        
#PBS -lfeature=ivybridge                                                                                                                                          
#PBS -t 1-80                                                                                                                                                      
index=${PBS_ARRAYID}
job=${PBS_JOBID}

rm -rf ~/.theano

module purge
module load numpy/intel/1.8.1
module load cython/intel/0.22
module load theano/20150721

$RUNDIR=$HOME/Divisive_Normalization_Project/Code/Demixing/

cd $RUNDIR
rm -rf ~/.theano

echo $index
THEANO_FLAGS="base_compiledir=$RUNDIR/$index/ " python demixing_gains.py $index

echo "Done"