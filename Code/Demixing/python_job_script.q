#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=2:00:00
#PBS -l mem=2GB
#PBS -N python
#PBS -M david.halpern@nyu.edu
#PBS -j oe
#PBS -t 1-8000
 
module purge
module load python

python demixing_gains.py $index

echo "Done"