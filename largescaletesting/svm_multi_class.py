from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import feature_matrix
import feature_vector
import ig_classes
import normalization
import shap
import matplotlib.pyplot as plt
import foldseek_feature
from sklearn.svm import LinearSVC

# Identify classes too small for stratified sampling
#pin_order = feature_matrix.read_pin_order("matched_results.csv", column="PIN")
pin_order = feature_matrix.read_pin_order("matched_and_merged_CD_HIT_90_final.csv", column="PIN")
print(len(pin_order))
# Separate features (X) from labels (y)
super_dict_features_only = {}
y_labels = {}
for pin, features in feature_vector.super_dict.items():
    features_copy = features.copy()
    label = features_copy.pop('Type', None)
    if label is not None:
        y_labels[pin] = label
    super_dict_features_only[pin] = features_copy
print(f"Initial number of samples: {len(y_labels)}")
filtered_super_dict = {}
filtered_y_labels = {}
for pin, label in y_labels.items():
    if label != "CD19":
        filtered_super_dict[pin] = super_dict_features_only[pin]
        filtered_y_labels[pin] = label
super_dict_features_only = filtered_super_dict
y_labels = filtered_y_labels
print(f"Number of samples after removing CD19: {len(y_labels)}")

# Now vectorize only the features
X, pins, feature_names = feature_matrix.vectorize_super_dict(
   super_dict_features_only,
   sort_pins=False,
   pin_order=pin_order,
   strict_order=True
)

# Create y array in the same order as pins
y = np.array([y_labels.get(pin) for pin in pins])
print(y)
print(y.shape)
#
## Normalize features
X = normalization.z_score_normalization(X) # 1052 x 66
foldseek = foldseek_feature.foldseek_matrix
X = np.hstack([X,foldseek])
print(foldseek.shape)
print(X.shape)
#X = np.concatenate((X.T[:26], X.T[62:67]))
X = X.T[67:]
#X = X.T[:6] 
X  = X.T
print(X.shape)
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Initial class counts:", class_counts)


too_small = {cls for cls, c in class_counts.items() if c < 2}
if too_small:
    print(f"\n⚠️ Removing classes with fewer than 2 samples: {too_small}")

    # Create a mask to keep only rows with acceptable classes
    mask = np.array([label not in too_small for label in y])

    # Apply the mask to both X and y to keep them aligned
    X = X[mask]
    y = y[mask]
    print(f"Number of samples after removing small classes: {len(y)}")

# Show final distribution
unique, counts = np.unique(y, return_counts=True)
print("\nFinal class counts after cleanup:", dict(zip(unique, counts)))
print("X shape:", X.shape, "y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        C=1.0,
        kernel="linear",
        probability=False,          # True gives softmax-like probabilities (slower)
        decision_function_shape="ovr",
        random_state=42
    ))
])

print("\n--- Performing 5-Fold Cross-Validation ---")

# Define the model pipeline

# Set up stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation to get accuracy scores
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

# Print the results
print(f"Cross-validation accuracy scores for each fold: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation of accuracy: {scores.std():.4f}")

# To get detailed predictions across all folds
y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

print("\n--- Cross-Validated Classification Report ---")
print(classification_report(y, y_pred_cv))

print("\n--- Cross-Validated Confusion Matrix ---")
print(confusion_matrix(y, y_pred_cv))
clf.fit(X_train, y_train)
print(f"Training set shape")
print(X_train.shape)
print(y_train.shape)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)
print("===Report for Train Set")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))


print(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


print(clf.named_steps["svc"].coef_.shape)
model = clf.named_steps["svc"]
class_names = clf.classes_
#explainer = shap.Explainer(model, X_train)
#shap_values = explainer(X_test)

#shap.summary_plot(shap_values, X_test, plot_type="bar")
def get_cv_coefs(model, X, y, cv):
    
    fold_coefs = []
    
    fold_idx = 1 
    for train_idx, val_idx in cv.split(X, y): 
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
    
        # Clone and train a fresh model for each fold
        model_fold = model
        model_fold.fit(X_tr, y_tr)

        coef_matrix = model_fold.coef_   # shape: (n_classes, n_features)
        fold_coefs.append(coef_matrix)

        print(f"\n--- Fold {fold_idx} Coefficients (shape {coef_matrix.shape}) ---")
        print(coef_matrix)
    
        fold_idx += 1
    
    print("\nCollected coefficients from all folds.")
    
    return fold_coefs

def plot_mean_coef_per_class(fold_coefs, class_names):
    coef_array = np.array(fold_coefs)   # (n_folds, n_classes, n_features)
    n_folds, n_classes, n_features = coef_array.shape

    # Compute mean across features for each fold
    # shape: (n_folds, n_classes)
    fold_means = np.mean(coef_array, axis=2)

    # Compute overall mean across folds: shape (n_classes,)
    class_avgs = np.mean(fold_means, axis=0)

    # Plot
    plt.figure(figsize=(14, 8))

    bar_width = 0.12
    x = np.arange(n_classes)

    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

    # Plot bars for each fold
    for f in range(n_folds):
        plt.bar(x + f * bar_width,
                fold_means[f],
                width=bar_width,
                label=f"Fold {f+1}",
                color=colors[f])

    # Plot average as a black line across each class group
    plt.plot(x + bar_width * (n_folds/2),
             class_avgs,
             color="black",
             marker="o",
             linestyle="-",
             linewidth=2,
             label="Average Across Folds")

    plt.xticks(x + bar_width * (n_folds/2), class_names, rotation=45, ha="right")
    plt.ylabel("Mean Coefficient Value")
    plt.title("Mean Coefficient per Class (per fold + average)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fold_means, class_avgs



fold_coefs = get_cv_coefs(clf.named_steps["svc"],X_train,y_train,cv)
print(f" Folds coef: {np.array(fold_coefs).shape}")
avg_coef = np.mean(np.array(fold_coefs), axis=0)                                                                                                                                               
print(f"Avg coef of the CV: {avg_coef}")                                                                                                                                                   
coef_var = np.var(np.array(fold_coefs), axis=0)                                                                                                                                            
print(f"Var of the coefs: {coef_var}")                                                                                                                                      
print(f"Mean of the Std: {np.mean(coef_var)}")
variance_per_class = np.mean(coef_var, axis=1)

print("\nVariance per class:")
#for i, v in enumerate(variance_per_class):
#   print(f"{class_names[i]}: {v}")



#class_means = plot_mean_coef_per_class(fold_coefs, class_names)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Get the trained Linear SVM model from the pipeline
svm = clf.named_steps["svc"]

# 2. Extract weights (for multiclass, will be shape: [n_classes, n_features])
weights = svm.coef_.flatten()

# 3. Fit a normal distribution
mu, std = norm.fit(weights)

# 4. Plot histogram
plt.hist(weights, bins=30, density=True, alpha=0.6)

# 5. Bell curve overlay
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, mu, std)
plt.plot(x, p, linewidth=3)

plt.title("Distribution of Linear SVM Weights with Fitted Gaussian")
plt.xlabel("Weight Value")
plt.ylabel("Density")
plt.show()

import numpy as np

# fold_coefs: shape (5, 9, 66)
# variance across folds (axis=0)
coef_var = np.var(fold_coefs, axis=0)

print(f"Coef variance shape:{coef_var.shape}")  # (9, 66)

import matplotlib.pyplot as plt

class_idx = 0  # choose class 0–8
for i in range(9):
    plt.figure(figsize=(14, 4))
    plt.bar(range(32), coef_var[class_idx])
    plt.xlabel("Feature index")
    plt.ylabel("Variance across folds")
    plt.title(f"Coefficient variance across folds – Class {class_names[class_idx]}")
    plt.tight_layout()
    plt.show()
    class_idx+=1


