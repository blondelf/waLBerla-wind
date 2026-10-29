#!/bin/bash
mpirun -np 6 TurbineLBM_CPU testCase.prm
python postprocess.py
