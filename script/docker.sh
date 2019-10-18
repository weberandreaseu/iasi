#!/bin/bash

docker build -t iasi .

docker run --rm -v "$(pwd):/iasi" -w /iasi --name iasi iasi \
    bash -c "source activate iasi && exec python -m unittest discover -v -s ./test"