import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


#creating 200 normal points and 10 anomalies
x§
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(200,2)
X_outliers = rng.uniform(low=-4, high = 4, size=(10,2)) #uniform distribution between -4 and 4 , has equal probability

#np.r_ is rowwise concatenation and np.c_ is columnwise concatenation
X = np.r_[X,X_outliers] # Joins X and X_outliers

clf = IsolationForest(n_estimators = 200,
                      max_samples='auto',
                      contamination=0.05,
                      random_state=42)

clf.fit(X)

labels = clf.predict(X)
scores = clf.decision_function(X)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1], c=labels, cmap = 'coolwarm', edgecolors='k')
plt.title("Isolation Forest Predictions\n1 = normal (blue), -1 = anomaly (red)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()