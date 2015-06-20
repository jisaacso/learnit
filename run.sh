#!/bin/bash

if [ ! -d "venv" ]
then
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

ECHO "Running a server at $1"
python learnit/server.py "$1"
