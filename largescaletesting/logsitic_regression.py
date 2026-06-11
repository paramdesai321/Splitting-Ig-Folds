from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.model_selection import train_test_split
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
#pin_order = feature_matrix.read_pin_order("matched_results.csv", column="PIN") # Original
updated_pin_order = feature_matrix.read_pin_order("matched_and_merged_CD_HIT_90_final.csv", column="PIN") # CD HIT 90
print(f"Updated PIN Order from matched and merge cd hit 90 final's length")
print(len(updated_pin_order))
# Separate features (X) from labels (y)
super_dict_features_only = {}
y_labels = {}
#for pin, features in feature_vector.super_dict.items(): # Original Snippet line 21-26
#    features_copy = features.copy()
#    label = features_copy.pop('Type', None)
#    if label is not None:
#        y_labels[pin] = label
#        print("hello")
#    super_dict_features_only[pin] = features_copy
for pin, features in feature_vector.super_dict.items(): # CD HIT 90 Snippet line 27-32
    #print(pin)
    features_copy = features.copy()
    print(features_copy)
    label = features_copy.pop('Type', None)
    #print(label)
    if label is not None:
        y_labels[pin] = label
        print("hello")
        #print(y_labels[pin])
    super_dict_features_only[pin] = features_copy
print(f'{len(y_labels)}')
print("LABELS")
filtered_super_dict = {}
filtered_y_labels = {}
for pin, label in y_labels.items():
    if label != "CD19":
        filtered_super_dict[pin] = super_dict_features_only[pin]
        filtered_y_labels[pin] = label
super_dict_features_only = filtered_super_dict
y_labels = filtered_y_labels
# Now vectorize only the features
#X, pins, feature_names = feature_matrix.vectorize_super_dict( # original 
#   super_dict_features_only,
#   sort_pins=False,
#   pin_order=pin_order,
#   strict_order=True
#)
#
X, pins, feature_names = feature_matrix.vectorize_super_dict(
   super_dict_features_only,
   sort_pins=False,
   pin_order=updated_pin_order,
   strict_order=True
)


print("Profile from Feature matrix")
print(f"Shape of : X={X.shape}, PDBS={len(pins)}, feature_names={len(feature_names)}")

# Create y array in the same order as pins
y = np.array([y_labels.get(pin) for pin in pins])
#print(y)
print(y.shape)
#
## Normalize features
X = normalization.z_score_normalization(X)
#print(X.shape)
#foldseek = foldseek_feature.foldseek_matrix
#print(foldseek.shape)
#X = np.hstack([X,foldseek])
print("Input shape")
print(X.shape)
index = 0

#X = X.T[index]
#X = X.T[-80:]
#X = X.T[:6] # Angle between Strands
#X = X.T[6:10] # Angle Between Strand Projections
#X  = X.T[[10, 13, 16,19], :]
#X  = X.T[[11, 14, 17,20], :]
#X  = X.T[[12, 15, 18,21], :] 
#X = X.T[22:27] # Distance to Center of Mass
#X= X.T[62:67] # Length of Strandsw
#X = X.T[26:63]  # Hydrophibicity
#X = X.T[67:] # Foldseek
#X = np.concatenate((X.T[:6], X.T[6:10],X.T[62:]))
X = np.concatenate((X.T[:26],X.T[62:67]))
#print("Checking feature values 27,28,29")
#print(X[27])
#print(X[28])
#print(X[29])
#X = X.T[66:]
X = X.T
#X = X.reshape(-1,1)
#print(f"Foldseek features")
#print(X)
#print(X.shape)
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
#print(np.isnan(X).any())
print("Unique labels:", np.unique(y))
nan_indices = np.argwhere(np.isnan(X))
#print(nan_indices)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- K-Fold Cross-Validation ---
print("\n--- Performing 5-Fold Cross-Validation ---")

# Define the model
model = LogisticRegression(penalty='l2',multi_class='multinomial', solver='lbfgs', max_iter=100) # Increased max_iter for convergence
# Set up stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation to get accuracy scores
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

# Print the results
print(f"Cross-validation accuracy scores for each fold: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation of accuracy: {scores.std():.4f}")

# To get a detailed classification report, we can use cross_val_predict
y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)

print("\n--- Cross-Validated Classification Report ---")
print(classification_report(y_train, y_pred_cv))

print("\n--- Cross-Validated Confusion Matrix ---")

print(confusion_matrix(y_train, y_pred_cv))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred_train = model.predict(X_train)
print("===Report for Train Set")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#def get_cv_coefs(model, X, y, cv):
#        
#    fold_coefs = []
#    
#    fold_idx = 1
#    for train_idx, val_idx in cv.split(X, y):
#        X_tr, X_val = X[train_idx], X[val_idx]
#        y_tr, y_val = y[train_idx], y[val_idx]
#        
#        # Clone and train a fresh model for each fold
#        model_fold = LogisticRegression(
#            penalty=model.penalty,
#            multi_class=model.multi_class,
#            solver=model.solver,
#            max_iter=model.max_iter
#        )
#        model_fold.fit(X_tr, y_tr)
#
#        coef_matrix = model_fold.coef_   # shape: (n_classes, n_features)
#        fold_coefs.append(coef_matrix)
#
#        print(f"\n--- Fold {fold_idx} Coefficients (shape {coef_matrix.shape}) ---")
#        print(coef_matrix)
#        
#        fold_idx += 1
#    
#    print("\nCollected coefficients from all folds.")
#    
#    return fold_coefs
#
#import numpy as np
#import matplotlib.pyplot as plt
#
#def plot_mean_coef_per_class(fold_coefs, class_names):
#    coef_array = np.array(fold_coefs)   # (n_folds, n_classes, n_features)
#    n_folds, n_classes, n_features = coef_array.shape
#    
#    # Compute mean across features for each fold
#    # shape: (n_folds, n_classes)
#    fold_means = np.mean(coef_array, axis=2)
#
#    # Compute overall mean across folds: shape (n_classes,)
#    class_avgs = np.mean(fold_means, axis=0)
#    
#    # Plot
#    plt.figure(figsize=(14, 8))
#    
#    bar_width = 0.12
#    x = np.arange(n_classes)
#
#    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
#
#    # Plot bars for each fold
#    for f in range(n_folds):
#        plt.bar(x + f * bar_width,
#                fold_means[f],
#                width=bar_width,
#                label=f"Fold {f+1}",
#                color=colors[f])
#    
#    # Plot average as a black line across each class group
#    plt.plot(x + bar_width * (n_folds/2),
#             class_avgs,
#             color="black",
#             marker="o",
#             linestyle="-",
#             linewidth=2,
#             label="Average Across Folds")
#
#    plt.xticks(x + bar_width * (n_folds/2), class_names, rotation=45, ha="right")
#    plt.ylabel("Mean Coefficient Value")
#    plt.title("Mean Coefficient per Class (per fold + average)")
#    plt.legend()
#    plt.grid(axis="y", alpha=0.3)
#    plt.tight_layout()
#    plt.show()
#
#    return fold_means, class_avgs
#
#
#print(f"Model coef: {model.coef_.shape}")
#import shap
#import color_igtype
#explainer = shap.Explainer(model, X)
#shap_values = explainer(X)
#
## Get the class names in the order that the model sees them
#class_names = model.classes_
#print(f"Model params: {model.coef_}")
#
#
## Get the feature names from the model
#n_features = model.coef_.shape[1]
#fold_coefs = get_cv_coefs(model,X_train,y_train,cv)
#fold_coefs = np.array(fold_coefs)
#print(f"Fold Coefs: {fold_coefs}")
#print(f"Fold Coefs Shape: {fold_coefs.shape}")
#avg_coef = np.mean(np.array(fold_coefs), axis=0)
#print(f"Avg coef of the CV: {avg_coef}")
#coef_var = np.var(np.array(fold_coefs), axis=0)
#print(f"Mean of the var: {np.mean(coef_var)}")
#variance_per_class = np.mean(coef_var, axis=1)
#
#print("\nVariance per class:")
#for i, v in enumerate(variance_per_class):
#    print(f"{class_names[i]}: {v}")
#
#
#class_means = plot_mean_coef_per_class(fold_coefs, class_names)
#print(coef_var.shape)  # (9, 66)
#
#
#plt.hist(model.coef_.flatten(), bins=20)
#plt.title("Distribution of Logistic Regression Weights")
#plt.show()
#
#from scipy.stats import norm
#
## Suppose these are your logistic regression weights:
#weights = model.coef_.flatten()
#
## Fit a normal distribution to the data
#mu, std = norm.fit(weights)
#
## Plot the histogram
#plt.hist(weights, bins=20, density=True, alpha=0.6)
#
## Create a smooth x range for the bell curve
#xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, 200)
#
## Compute the PDF of the fitted Gaussian
#p = norm.pdf(x, mu, std)
#
## Plot the bell curve
#plt.plot(x, p, linewidth=3)
#
#plt.title("Weight Distribution with Fitted Gaussian")
#plt.xlabel("Weight Value")
#plt.ylabel("Density")
#plt.show()
#
#import numpy as np
#
## fold_coefs: shape (5, 9, 66)
## variance across folds (axis=0)
#coef_var = np.var(fold_coefs, axis=0)
#
#print(coef_var.shape)  # (9, 66)
#
#import matplotlib.pyplot as plt
##
##class_idx = 0  # choose class 0–8
##for i in range(9):
##    plt.figure(figsize=(14, 4))
##    plt.bar(range(32), coef_var[class_idx])
##    plt.xlabel("Feature index")
##    plt.ylabel("Variance across folds")
##    plt.title(f"Coefficient variance across folds – Class {class_names[class_idx]}")
##    plt.tight_layout()
##    plt.show()
##    class_idx+=1
##
##
##final_feature_names = [feature_names[i] for i in indices]
#
##final_feature_names = feature_names[index]
##final_feature_names = feature_names[index]
##print(final_feature_names)
##print(feature_names[-4:])
##print(feature_names[index])
##shap.summary_plot(shap_values, X, plot_type="bar", class_names=class_names, feature_names=final_feature_names)
##
##w = model.coef_.ravel()
##
### Two random directions in parameter space
##d1 = np.random.randn(*w.shape)
##d2 = np.random.randn(*w.shape)
##
### Normalize directions (for scaling)
##d1 /= np.linalg.norm(d1)
##d2 /= np.linalg.norm(d2)
##
### Create grid
##alphas = np.linspace(-2, 2, 30)
##betas = np.linspace(-2, 2, 30)
##loss_surface = np.zeros((len(alphas), len(betas)))
##
### Compute loss at each grid point
##for i, a in enumerate(alphas):
##    for j, b in enumerate(betas):
##        w_perturbed = w + a * d1 + b * d2
##        model_pert = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1)
##        model_pert.coef_ = w_perturbed.reshape(model.coef_.shape)
##        model_pert.intercept_ = model.intercept_
##        y_pred_proba = model.predict_proba(X)
##        loss_surface[i, j] = log_loss(y, y_pred_proba)
##
### Plot
##from mpl_toolkits.mplot3d import Axes3D
##
##fig = plt.figure(figsize=(8,6))
##ax = fig.add_subplot(111, projection='3d')
##A, B = np.meshgrid(alphas, betas)
##ax.plot_surface(A, B, loss_surface, cmap='viridis')
##ax.set_xlabel('Direction 1 (α)')
##ax.set_ylabel('Direction 2 (β)')
##ax.set_zlabel('Log Loss')
##plt.title('3D Loss Landscape (Logistic Regression)')
##plt.show()
#import numpy as np
#
#def feature_range_by_class(
#    clf,
#    X,
#    feature_idx,
#    feature_name=None,
#    n_points=300,
#    means=None,
#    stds=None
#):
#    """
#    Sweep one feature and return ranges where each class is predicted.
#
#    Parameters
#    ----------
#    clf : trained sklearn LogisticRegression
#    X : ndarray, shape (n_samples, n_features)  (RAW features)
#    feature_idx : int
#        Index of feature to sweep
#    feature_name : str (optional)
#    n_points : int
#        Resolution of sweep
#    means, stds : ndarray or None
#        If model was trained on standardized data, pass training mean/std
#
#    Returns
#    -------
#    dict : {class_index: [(start, end), ...]}
#    """
#
#    # Sweep values in RAW feature space
#    x_vals = np.linspace(
#        X[:, feature_idx].min(),
#        X[:, feature_idx].max(),
#        n_points
#    )
#
#    # Reference point: mean of raw data
#    X_ref = np.tile(X.mean(axis=0), (n_points, 1))
#    X_ref[:, feature_idx] = x_vals
#
#    # Standardize if needed
#    if means is not None and stds is not None:
#        X_model = (X_ref - means) / stds
#    else:
#        X_model = X_ref
#
#    # Predict classes
#    probs = clf.predict_proba(X_model)
#    preds = probs.argmax(axis=1)
#
#    n_classes = probs.shape[1]
#    ranges = {c: [] for c in range(n_classes)}
#
#    # Extract contiguous intervals
#    for c in range(n_classes):
#        mask = preds == c
#        start = None
#
#        for i, m in enumerate(mask):
#            if m and start is None:
#                start = x_vals[i]
#            elif not m and start is not None:
#                ranges[c].append((start, x_vals[i - 1]))
#                start = None
#
#        if start is not None:
#            ranges[c].append((start, x_vals[-1]))
#
#    # Pretty print
#    fname = feature_name if feature_name else f"feature_{feature_idx}"
#    print(f"\nFeature sweep: {fname}")
#
#    for c, intervals in ranges.items():
#        if intervals:
#            print(f"Class {c}:")
#            for lo, hi in intervals:
#                print(f"  [{lo:.3f}, {hi:.3f}]")
#
#    return ranges
#
## If you trained on standardized features:
#means = X_train.mean(axis=0)
#stds = X_train.std(axis=0)
#
#feature_range_by_class(
#    clf=model,
#    X=X_train,      # raw (unscaled) data
#    feature_idx=0,
#    feature_name="angle_deg",
#    means=means,
#    stds=stds
#)
#
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_text
#
#tree = DecisionTreeClassifier(max_depth=6)
#tree.fit(X_train, model.predict(X_train))
#feature_names = np.array(feature_names)
#feature_names=  np.concatenate((feature_names[:28], feature_names[62:]))
#print(export_text(tree, feature_names=feature_names))
#
#from sklearn.tree import plot_tree
#import matplotlib.pyplot as plt
#
#plt.figure(figsize=(50, 50))  # adjust size if needed
#plot_tree(
#    tree,
#    feature_names=feature_names,
#    class_names=tree.classes_,
#    filled=True,
#    rounded=True,
#    fontsize=8
#)
#
#plt.title("Decision Tree for Ig Fold Classification")
#plt.show()
