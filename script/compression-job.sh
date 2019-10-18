#!/bin/bash#SBATCH --nodes=1
#SBATCH --mem=20000
#SBATCH --cpus-per-task=20
#SBATCH --time=36:00:00
#SBATCH --job-name iasi_compression_NUM

printf "[%s] %s\n" "$(date +'%D %T')" "start iasi-compression"

module load devel/python/3.6

YEAR=2016

project=${PROJECT}/iasi


# create a workspace and replace 'iasi-compression' with yours
DST=$(ws_find iasi-compression)
SRC=$(ws_find iasi-compression)/retrieval/$YEAR


# it re recommended to disable multithreading in numpy
# instead, let each core processes it's own file
# this should be more efficient since it scales linear with number of cores
# see "--workers $SLURM_CPUS_PER_TASK" below
NUM_THREADS=1
export MKL_NUM_THREADS=$NUM_THREADS
export NUMEXPR_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

cd "$project" || exit
source activate iasi


# process date range of orbits
# pass the date range as an argument $1 to the script e.g.
# --date-interval 2019 
# --date-interval 2019-05
# --date-interval 2019-05-03-2019-06-19
# for details see https://luigi.readthedocs.io/en/stable/_modules/luigi/date_interval.html
python -m luigi --module iasi.compression CompressDateRange \
        --date-interval "$1" \
        --src "$SRC" \
        --dst "$DST" \
        --workers "$SLURM_CPUS_PER_TASK" \
        --no-lock # prevents exclusive assignment to a node in case of multiple compute nodes

# python -m luigi --module iasi CompressDataset \
#       --log-file \
#       --dst $DST \
#       --file $DST/retrieval/2016/METOPA_20160215103553_48385_20190323002832.nc
#       --workers $SLURM_CPUS_PER_TASK