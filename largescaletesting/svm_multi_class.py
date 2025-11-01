from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import feature_matrix
import feature_vector
import ig_classes
import normalization
pin_order = feature_matrix.read_pin_order("consolidated_pdbs_umesh_noCD19.csv", column="PIN")

#X = feature_matrix.X.T
#X = X[:1048]
X  = normalization.z_normalized_features.T
print(X.shape)
types_list = [vals.get("Type") for vals in feature_vector.super_dict.values()]
y = types_list
print(len(y))
print(y)
y = np.array(y)
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
print("\nFinal class counts after cleanup:", dict(zip(unique, counts)))
print("X shape:", X.shape, "y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)


clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

