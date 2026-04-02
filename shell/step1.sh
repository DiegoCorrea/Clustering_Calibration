#!/bin/bash
cd ./pierre_in_frame &&
#python3 step1_preprocessing.py from_file=YES file_name=ml-1m_cross &&
#python3 step1_preprocessing.py from_file=YES file_name=ml-1m_class_distribution_cross &&
#python3 step1_preprocessing.py from_file=YES file_name=ml-1m_distribution_cross &&
#python3 step1_preprocessing.py from_file=YES file_name=ml-1m_ohe_cross &&
python3 step1_preprocessing.py from_file=YES file_name=yahoo_movies_cross &&
python3 step1_preprocessing.py from_file=YES file_name=yahoo_movies_class_distribution_cross &&
python3 step1_preprocessing.py from_file=YES file_name=yahoo_movies_distribution_cross &&
python3 step1_preprocessing.py from_file=YES file_name=yahoo_movies_ohe_cross