# Import required libraries
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

# Create a dataset with a single feature
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)

# Perform feature transformation using binning
binning = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_binned = binning.fit_transform(X)

# Print original and transformed features
print("Original feature:")
print(X)
print("Transformed feature:")
print(X_binned)

#This code creates a dataset with a single feature, performs feature transformation using binning with 3 bins of equal width, and prints the original and transformed features. The KBinsDiscretizer class from scikit-learn is used to perform the binning feature transformation.
#You can modify this code to work with other datasets, use a different number of bins and binning strategy, and use a different feature transformation technique as needed.
