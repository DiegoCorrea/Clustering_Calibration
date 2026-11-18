#!/bin/bash
# Re-generates the calibrated lists after the step-1 fix (GENRE vs. GENRES distribution class).
cd ./pierre_in_frame &&
python3 step4_postprocessing.py from_file=YES file_name=movies_cross_genre_fix
