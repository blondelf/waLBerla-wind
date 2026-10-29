#!/bin/bash
mpirun -np 6 TurbineLBM_CPU_CP testCase.prm
python postprocess.py
