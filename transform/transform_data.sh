#!/bin/bash

python transform/filter_articles.py
python transform/filter_sentences.py
python transform/finalize.py
