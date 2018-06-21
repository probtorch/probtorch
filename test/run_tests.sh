#!/usr/bin/env bash
pushd $(dirname "$0") > /dev/null

#for f in ./test_*.py
for f in test_concrete.py test_stochastic.py
do
    PYTHONPATH='../' python "$f"
done
popd > /dev/null
