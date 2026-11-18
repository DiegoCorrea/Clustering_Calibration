#!/bin/bash
# Re-execution of the experiment for the revision (Reviewer 3), after the fix of the genre distribution class in
# step 1. Steps 2 (recommender search) and 3 (candidate items) were not affected and are not re-executed.
# Run from the project root. The original (incorrect) state is preserved in data.zip and results.zip.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd ./pierre_in_frame &&
# Step 1: genre preference distributions (only the DISTRIBUTION option)
python3 step1_preprocessing.py from_file=YES file_name=yahoo_movies_distribution_cross &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_distribution_cross &&
python3 step1_preprocessing.py from_file=YES file_name=twitter_movies_distribution_cross &&
# Step 2: selection of the labeling models (CONFORMITY option only)
python3 step2_searches.py from_file=YES file_name=movies_cross_conformity_genre_fix &&
# Step 4: calibrated lists (equivalent to: python3 step4_postprocessing.py from_file=YES file_name=movies_cross_genre_fix)
python3 ../shell/step4_genre_fix_ordered.py &&
# Step 5: calibration check, stage-wise protocol, scalability benchmark
python3 step5_metrics.py from_file=YES file_name=movies_cross_calibration_check &&
python3 step5_metrics.py from_file=YES file_name=movies_cross_stagewise &&
python3 step5_metrics.py from_file=YES file_name=twitter_movies_cross_scalability &&
# Step 6: compilation and statistical tests (results/stagewise)
python3 step6_protocol.py from_file=YES file_name=movies_cross_stagewise &&
# Step 7: figures and LaTeX tables (results/paper_revision)
python3 step7_charts_tables.py from_file=YES file_name=movies_stagewise &&
python3 ../results/paper_revision/build_paper.py
