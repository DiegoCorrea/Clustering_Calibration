#!/bin/bash
cd ./pierre_in_frame &&
python3 step7_charts_tables.py from_file=YES file_name=ml-1m &&
python3 step7_charts_tables.py from_file=YES file_name=ml-1m_conformity &&
python3 step7_charts_tables.py from_file=YES file_name=yahoo_movies &&
python3 step7_charts_tables.py from_file=YES file_name=yahoo_movies_conformity &&
python3 step7_charts_tables.py from_file=YES file_name=twitter_movies &&
python3 step7_charts_tables.py from_file=YES file_name=twitter_movies_conformity