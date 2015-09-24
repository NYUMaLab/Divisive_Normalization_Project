#!/bin/bash                                                                                                                                                                                                                                                                                                                        
#PBS -l nodes=1:ppn=4                                                                                                                                                                                                                                                                                                              
#PBS -l walltime=5:00:00                                                                                                                                                                                                                                                                                                           
#PBS -l mem=2GB                                                                                                                                                                                                                                                                                                                    
#PBS -N python                                                                                                                                                                                                                                                                                                                    
#PBS -M david.halpern@nyu.edu                                                                                                                                                                                                                                                                                             
#PBS -j oe                                                                                                                                                                                                                                                                                                                        
#PBS -lfeature=ivybridge                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                  
index=${PBS_ARRAYID}
job=${PBS_JOBID}

module purge
module load numpy/intel/1.8.1
module load cython/intel/0.22
module load theano/20150721

RUNDIR=$HOME/Divisive_Normalization_Project/Code/Demixing/

cd $RUNDIR
rm -rf ~/.theano

THEANO_FLAGS="base_compiledir=$RUNDIR/compiledir/$index/ " python demixing_part0b.py 0

echo "Done"