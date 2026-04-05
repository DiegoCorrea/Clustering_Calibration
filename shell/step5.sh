#!/bin/bash
cd ./pierre_in_frame &&

python3 step5_metrics.py from_file=YES file_name=ml-1m_cross &&
python3 step5_metrics.py from_file=YES file_name=ml-1m_cross_cross_conformity &&
python3 step5_metrics.py from_file=YES file_name=yahoo_movies_cross &&
python3 step5_metrics.py from_file=YES file_name=yahoo_movies_cross_conformity