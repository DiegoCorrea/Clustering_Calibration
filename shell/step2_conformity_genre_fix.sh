#!/bin/bash
# Re-selects the labeling-model configurations on the (fixed) genre preference distributions.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ./pierre_in_frame &&
python3 step2_searches.py from_file=YES file_name=movies_cross_conformity_genre_fix
