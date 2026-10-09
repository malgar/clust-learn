Replication code
===================================================

These files reproduce the illustrative example in the code paper end-to-end.

Requirements
------------
Python 3.10. Use the pinned `requirements.txt`:

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
The results reported in the paper were produced in the Code Ocean capsule
(Python 3.10, with the package versions pinned in requirements.txt):
12.0% missing values imputed to a complete set of 4,556 records; 4,242 records
after outlier removal; 16 derived components (15 numerical via SPCA + 1
categorical via MCA); k-means with k=6 selected by the elbow method; XGBoost
cluster classifier with 83.51% test accuracy and ESCS as the dominant SHAP
predictor. Results may differ with other Python or library versions.
