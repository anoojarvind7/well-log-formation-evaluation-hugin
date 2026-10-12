.PHONY: env data core run test all clean

PY ?= python

env:            ## create the conda environment
	conda env create -f environment.yml

data:           ## download FORCE 2020 logs (md5-verified) into data/raw/
	$(PY) scripts/fetch_data.py

core:           ## also download the core plug xlsx from Zenodo
	$(PY) scripts/fetch_data.py --core

run:            ## steps 1-11 -> results/
	$(PY) -m hugin.pipeline

test:
	$(PY) -m pytest -q

all: data run test

clean:          ## remove cached intermediate data (raw data is kept)
	rm -rf data/interim
