SHELL := /usr/bin/env bash

.PHONY: install train test

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python src/train.py

test:
	python src/validate.py
