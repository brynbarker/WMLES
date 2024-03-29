#!/bin/bash

(cd $MAIN/pyranda/ && python setup.py clean)
(cd $MAIN/pyranda/ && python setup.py install)
