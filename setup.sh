#!/bin/bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
python -m spacy download en_core_web_sm
