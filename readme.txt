1) Create a virtual environment using Python 3.9
2) Install numpy
pip install -U numpy
3) Install other requirements
pip install -r requirements.txt

DESCRIPTION:
This project contains the implementation of the IPDD adaptive for the control-flow perspective.
The implementation was firtly validated by this project, and then, integrated into the IPDD framework.
All the experiments reported in the Thesis "Concept Drift in Process Models" can be reproduced using the
script execute_experiments.py. We also saved the results from IPDD in the folder
"experiments_results/IPDD_controlflow_adaptive".

The script analyze_metrics_experiments.py creates the plots available in the thesis.

The Apromore results are saved in the folder "experiments_results/Apromore".

For VDD System, we only saved the compiled metrics due to space issues. The output logs of VDD for the two
tested datasets generate 112Gb of data.

For calculating the metrics F-score, Mean Delay, and FPR (False Positive Rate):
1) Read the output files from VDD and Apromore and format into a compiled excel file: compile_results_ProDrift.py
and compile_results_VDD.py --> results_XXX.xlsx
2) Run the script calculate_metrics_experiments.py. This script reads the result file (excel) and create a new excel
file with the calculated metrics

