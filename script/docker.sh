#!/bin/bash

docker run --rm -v $(pwd):/iasi -w /iasi --name iasi jupyter/scipy-notebook \
    pip install -r requirements.txt && \
    python -m unittest discover -v -s ./test