#!/bin/bash

# used to reduce numpy threads for task level parallelism

NUM_THREADS=1

export MKL_NUM_THREADS=$NUM_THREADS
export NUMEXPR_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS
