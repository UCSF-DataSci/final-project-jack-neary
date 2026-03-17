<!-- brew install google-cloud-sdk -->
# ECG-Based ICU Mortality Prediction
 
## Overview
 
This project investigates which ECG features are most predictive of in-hospital mortality among ICU patients using the MIMIC-IV clinical database. We build and compare multiple machine learning models — Logistic Regression, Random Forest, Gradient Boosting, and XGBoost — trained on ECG measurements and patient demographics extracted from BigQuery.
 
The core research question is:
 
> **Among ICU patients with ECG recordings, which ECG features are most predictive of in-hospital mortality?**
 
---

## Dataset

### Input Features

### Target Variable

### Dataset Dimensions


## How to Run


## Decisions & Trade-offs


## Example Output


## Citations
 
### Data
- Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58. 
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. 
- Gow, B., Pollard, T., Nathanson, L. A., Johnson, A., Moody, B., Fernandes, C., Greenbaum, N., Waks, J. W., Eslami, P., Carbonati, T., Chaudhari, A., Herbst, E., Moukheiber, D., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/4nqg-sb35. 
- https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/measurement (adapted vitals_query from here)


