#!/bin/bash

deactivate || echo "we're ready"

rm -rd $MAIN/.venvs/venv2
echo $(ls $MAIN/.venvs)
virtualenv -p python2 $MAIN/.venvs/venv2

vactivate2
python -m pip install numpy scipy mpi4py h5py matplotlib

python -m pip list

(cd $MAIN/pyranda/ && python setup.py install)

echo "venv2 ready for use"
