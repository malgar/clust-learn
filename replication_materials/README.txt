Replication code
===================================================

These files are drop-in replacements for the repository's `replication_materials/`.
They reproduce the illustrative example in the code paper end-to-end.

Requirements
------------
Python 3.9-3.11. Use the pinned `requirements.txt`:

    pip install -r requirements.txt

On macOS, xgboost additionally needs the OpenMP runtime:

    brew install libomp

How to run
----------
    1. Open a command prompt.
    2. cd to the directory where paper_script.py is located.
    3. Run:  python paper_script.py

Numerical outputs (tables / data frames) are printed to the console; figures are
saved to a directory called "img" (created automatically).

Input data is in the "data" folder (pisa_spain_sample_v2.csv), also available at
https://github.com/malgar/clust-learn/tree/master/notebooks/data

Reproducibility note
--------------------
With the pinned environment above and the fixed random seeds in the script, the run
is deterministic. Representative outputs: 12.0% missing values imputed to a complete
set of 4,556 records; 264 outliers removed (4,292 retained); 15 derived components
(14 numerical via SPCA + 1 categorical via MCA); k-means with k=8 selected by the
elbow method; XGBoost cluster classifier with ~0.80 test accuracy and ESCS as the
dominant SHAP predictor. Exact figures depend on the pinned library versions.
