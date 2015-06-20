#!/bin/bash

if [ ! -d "venv" ]
then
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python learnit/test.py "$1" "$2"
