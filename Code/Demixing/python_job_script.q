#!/bin/bash                                                                     
#PBS -l nodes=1:ppn=4                                                           
#PBS -l walltime=2:00:00                                                        
#PBS -l mem=2GB                                                                 
#PBS -N python                                                                  
#PBS -M david.halpern@nyu.edu                                                   
#PBS -j oe                                                                      

module purge
module load theano

RUNDIR=$SCRATCH/demixing
mkdir -p $RUNDIR
cp $HOME/Divisive_Normalization_Project/Code/Demixing/* $RUNDIR

cd $RUNDIR
for i in {1..60} ; do
   echo $i
   python demixing_gains.py $i
done