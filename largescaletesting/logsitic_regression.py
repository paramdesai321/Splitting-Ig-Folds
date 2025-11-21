from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import feature_matrix
import ig_classes
import feature_vector
import numpy as np
import normalization
import main
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import foldseek_feature
pin_order = feature_matrix.read_pin_order("matched_results.csv", column="PIN")
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

# Now vectorize only the features
X, pins, feature_names = feature_matrix.vectorize_super_dict(
   super_dict_features_only,
   sort_pins=False,
   pin_order=pin_order,
   strict_order=True
)

# Create y array in the same order as pins
y = np.array([y_labels.get(pin) for pin in pins])
#print(y)
print(y.shape)
#
## Normalize features
X = normalization.z_score_normalization(X)
foldseek = foldseek_feature.foldseek_matrix
X = np.hstack([X,foldseek])
print("Input shape")
print(X.shape)
index = 0

#X = X.T[index]
X = X.T[-80:]
#X = X.T[:6]
#X = X.T[6:10]
#X  = X.T[[10, 13, 16,19], :]
#X  = X.T[[11, 14, 17,20], :]
#X  = X.T[[12, 15, 18,21], :]
#X = X.T[22:26]
#X= X.T[-4:]
#X = X.T[26:62]
#X = X.T[:63]  # Hydrophibicity
#X = np.concatenate((X.T[:28], X.T[62:]))
X = X.T
#X = X.reshape(-1,1)
#print(X)
print(X.shape)
#
#print(X.shape)
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


# --- K-Fold Cross-Validation ---
print("\n--- Performing 5-Fold Cross-Validation ---")

# Define the model
model = LogisticRegression(penalty='l2',multi_class='multinomial', solver='lbfgs', max_iter=1000) # Increased max_iter for convergence
# Set up stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation to get accuracy scores
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Print the results
print(f"Cross-validation accuracy scores for each fold: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation of accuracy: {scores.std():.4f}")

# To get a detailed classification report, we can use cross_val_predict
y_pred_cv = cross_val_predict(model, X, y, cv=cv)

print("\n--- Cross-Validated Classification Report ---")
print(classification_report(y, y_pred_cv))

print("\n--- Cross-Validated Confusion Matrix ---")

model.fit(X,y)
print(confusion_matrix(y, y_pred_cv))
print(f"Model coef: {model.coef_.shape}")
import shap
import color_igtype
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Get the class names in the order that the model sees them
class_names = model.classes_

# Get the feature names from the model
n_features = model.coef_.shape[1]
#final_feature_names = feature_names[[10, 13, 16,19], :].to_list
#final_feature_names = [feature_names[i] for i in indices]

#final_feature_names = feature_names[index]
final_feature_names = feature_names[index]
#print(final_feature_names)
#print(feature_names[-4:])
print(feature_names[index])
shap.summary_plot(shap_values, X, plot_type="bar", class_names=class_names, feature_names=final_feature_names)
#
#w = model.coef_.ravel()
#
## Two random directions in parameter space
#d1 = np.random.randn(*w.shape)
#d2 = np.random.randn(*w.shape)
#
## Normalize directions (for scaling)
#d1 /= np.linalg.norm(d1)
#d2 /= np.linalg.norm(d2)
#
## Create grid
#alphas = np.linspace(-2, 2, 30)
#betas = np.linspace(-2, 2, 30)
#loss_surface = np.zeros((len(alphas), len(betas)))
#
## Compute loss at each grid point
#for i, a in enumerate(alphas):
#    for j, b in enumerate(betas):
#        w_perturbed = w + a * d1 + b * d2
#        model_pert = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1)
#        model_pert.coef_ = w_perturbed.reshape(model.coef_.shape)
#        model_pert.intercept_ = model.intercept_
#        y_pred_proba = model.predict_proba(X)
#        loss_surface[i, j] = log_loss(y, y_pred_proba)
#
## Plot
#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111, projection='3d')
#A, B = np.meshgrid(alphas, betas)
#ax.plot_surface(A, B, loss_surface, cmap='viridis')
#ax.set_xlabel('Direction 1 (α)')
#ax.set_ylabel('Direction 2 (β)')
#ax.set_zlabel('Log Loss')
#plt.title('3D Loss Landscape (Logistic Regression)')
#plt.show()
