from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import feature_matrix
import ig_classes
import feature_vector
import numpy as np
import normalization
import main
pin_order = feature_matrix.read_pin_order("consolidated_pdbs_umesh_noCD19.csv", column="PIN")

X, pins, feature_names = feature_matrix.vectorize_super_dict(
    feature_vector.super_dict,
    sort_pins=False,
    pin_order=pin_order,
    strict_order=True
)
print(f"X.shape")
# X has shape (n_samples, n_features)
# y must be created with the same sample order
y = np.array([feature_vector.super_dict[pin].get("Type") for pin in pins])

# Normalize features
X = normalization.z_score_normalization(X)

print(X.shape)
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Initial class counts:", class_counts)

# Identify classes too small for stratified sampling
too_small = {cls for cls, c in class_counts.items() if c < 2}
if too_small:
    print(f"\n⚠️ Removing classes with fewer than 2 samples: {too_small}")

    # Create a mask to keep only rows with acceptable classes
    mask = np.array([label not in too_small for label in y]) 

    # Apply the mask to both X and y to keep them aligned
    X = X[mask]
    y = y[mask]


# Show final distribution
unique, counts = np.unique(y, return_counts=True)
#print("\nFinal class counts after cleanup:", dict(zip(unique, counts)))
print("X shape:", X.shape, "y shape:", y.shape)
print(np.isnan(X).any())
nan_indices = np.argwhere(np.isnan(X))
print(nan_indices)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)




model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
