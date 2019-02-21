#!/bin/bash
pwd
whoami
ls -la
ls -la data
ls -la tmp
python -m venv /tmp/venv
ls -la tmp
ls -la tmp/venv
source /tmp/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt