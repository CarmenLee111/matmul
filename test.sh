#!/bin/bash -l

#SBATCH -A g2019005
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=carmen.jing.lee@gmail.com
#SBATCH -p node -N 2 -n 25
#SBATCH -t 10:00

module load gcc openmpi

##############################################################
# input files    |     cores allowed
# input3600.txt  | 4, 9, 16, 25, 36, 64, 81, 100, 144, 225,...
# input5716.txt  | 4, 16, 1429,...
# input7488.txt  | 4, 9, 16, 36, 64, 81, 144, 169,...
# input9072.txt  | 4, 9, 16, 36, 49, 64, 81, 144, 196,...
# input10525.txt | 25, 421, 625,...
##############################################################
path=/proj/g2019005/nobackup/matmul_indata/
executable=matmul

c=25 

for f in input5.txt
do
    for i in 1 2 3
    do
        echo Number of cores $c
        echo input file name: $f
        mpirun -np $c ./$executable $path$f outputfile.txt
    done
done



