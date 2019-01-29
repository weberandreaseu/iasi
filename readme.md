# IASI

[![Build Status](https://travis-ci.com/weberandreaseu/iasi.svg?branch=master)](https://travis-ci.com/weberandreaseu/iasi)

## Setup

1. Create virtual environment: `python -m venv venv`
1. Activate environment: `source venv/bin/activate`
1. Install dependencies: `pip install -r requirements.txt`
1. Run unittests: `python -m unittest discover -v -s ./test`


## Usage

To manage processing of files, this project uses [Luigi](https://github.com/spotify/luigi/).
Processing steps are implemented as [Luigi Tasks](https://luigi.readthedocs.io/en/stable/tasks.html).
To start the luigi server run:
```
luigid
```
The backend should be available at [http://localhost:8082/](http://localhost:8082/).

To schedule a task e.g. `DeltaDRetrieval` run following command with your custom arguments:

```
python -m luigi --module iasi DeltaDRetrieval \
    --file ./test/resources/IASI-test-single-event.nc \
    --dst ./data \
    --svd 
```

For further details have a look at the [Luigi Documentation](https://luigi.readthedocs.io/).