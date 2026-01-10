# ML Audit Library
**Solves Data Lineage Blindness by tracking granular preprocessing steps.**

ml-audit is a lightweight Python library designed to bring transparency and reproducibility to data preprocessing. Unlike standard experiment trackers that treat preprocessing as a black box, this library records every granular transformation applied to your pandas DataFrame.


![Image](/static/uploads/content/ml_audit_preview_1768026935.png)

### 📌Key Features

* Full Audit Trail: Automatically logs every step (Imputation, Scaling, Encoding, etc.) into a JSON audit file.
* Reproducibility: Verify if your data pipeline produces the exact same result every time using hash validation.
* Visualization: Auto-generates an interactive HTML timeline of your preprocessing steps.


<a href="https://github.com/SHIVOGOJOHN/ml_audit" class="btn" target="_blank" style="display: inline-block; margin: 5px;"><i class="fab fa-github"></i> View On Github</a> <a href="https://pypi.org/project/ml-audit/" class="btn" target="_blank" style="display: inline-block; margin: 5px;"><i class="fas fa-box-open"></i> View on PyPI</a> <a href="https://www.mlaudit.info/" class="btn" target="_blank" style="display: inline-block; margin: 5px;"><i class="fas fa-book"></i>View Documentation</a> 