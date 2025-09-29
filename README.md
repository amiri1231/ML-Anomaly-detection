# Network Traffic Anomaly Detection using Machine Learning

This project builds an end to end pipeline to detect anomalies in network traffic flows using unsupervised learning.  
It is implemented in Python and trained on datasets from the CIC IDS 2017 intrusion detection dataset, it uses Isolation Forest to detect anomaly changes and outliers



#  Features
- Data Collection:
    - from CSV flow datasets (CIC-IDS 2017).

- Feature preparation:
  - Drop identifiers (IP, ports, timestamps).
  - Keep numeric features only.
  - Custom engineered features:
    - Forward/Backward packet ratio
    - Forward/Backward bytes ratio
    - Bytes per second (throughput)
    - Packets per second (rate)
- Modeling:
  - Isolation Forest for anomaly detection (scikit-learn).
  - Maybe PyOD in future.
- Evaluation:
  - Stratified train/validation/test split.
  - Metrics: AUC-PR, F1, Precision, Recall, Confusion Matrix.
  - Threshold chosen on validation to maximize F1.




