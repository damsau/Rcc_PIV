#! /bin/bash

work_dir="/home/masuda/01_github/10_Rcc_PIV"
params_file="${work_dir}/parameters/default.yaml"
python ${work_dir}/rcc_piv.py --params-file $params_file 
# python ${work_dir}/test.py --params-file $params_file 


