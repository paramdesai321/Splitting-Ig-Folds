import matplotlib.pyplot as plt
import numpy as np
import feature_matrix

def plot_violin(data, title, xticklabels, xlabel, ylabel):
    """
    Generates a violin plot for the given data.

    Args:
        data: A list of arrays, where each array contains the data for one violin.
        title: The title of the plot.
        xticklabels: The labels for the x-axis ticks.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
    """
    fig, ax = plt.subplots()
    print(np.argmin(data))
    print(np.argmax(data))
    ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, len(xticklabels) + 1))
    ax.set_xticklabels(xticklabels)
    plt.show()

def get_angle_between_strands_data():
    """
    Extracts the angle between strands data from the feature matrix.
    """
    X = feature_matrix.X
    # The first 6 features are the angles between strands
    data = [X[i] for i in range(6)]
    # Convert to degrees and remove NaNs
    data = [d[~np.isnan(d)] * 180 / np.pi for d in data]
    xticklabels = ["B and C", "B and E", "B and F", "C and E", "C and F", "E and F"]
    return data, "Angle between strands", xticklabels, "Strand Pairs", "Angle (degrees)"

def get_projections_data():
    """
    Extracts the projections data from the feature matrix.
    """
    X = feature_matrix.X
    # Features from index 5 to 8 are projections
    data = [X[i] for i in range(6, 9)]
    # Convert to degrees and remove NaNs
    data = [d[~np.isnan(d)] * 180 / np.pi for d in data]
    xticklabels = ["B and C", "B and F", "C and E", "C and F"] # Placeholder labels
    return data, "Projections", xticklabels, "Projection", "Angle(degrees)"

def get_distances_data():
    """
    Extracts the distances data from the feature matrix.
    """
    X = feature_matrix.X
    # The last 4 features are distances
    data = [X[i] for i in range(-4, 0)]
    data = [d[~np.isnan(d)] for d in data]
    #print(data.shape)
    
    xticklabels = ["B", "C", "E", "F"] # Placeholder labels
    return data, "Distances", xticklabels, "Distance", "Value"

def get_x_axis_data():
    """
    Extracts the x-axis data from the feature matrix.
    Plots features at indices 10, 13, 16, ...
    """
    X = feature_matrix.X
    num_features = X.shape[0]
    indices = np.arange(10, num_features, 3)
    indices = indices[:4]
    data = [X[i] for i in indices]
    data = [d[~np.isnan(d)] for d in data]
    xticklabels = ['B','C','E','F']
    return data, "X-axis Data", xticklabels, "Axis", "Value"

def get_y_axis_data():
    """
    Extracts the y-axis data from the feature matrix.
    Plots features at indices 11, 14, 17, ...
    """
    X = feature_matrix.X
    num_features = X.shape[0]
    indices = np.arange(11, num_features, 3)
    indices = indices[:4]
    data = [X[i] for i in indices]
    data = [d[~np.isnan(d)] for d in data]
    xticklabels = ['B','C','E','F']
    return data, "Y-axis Data", xticklabels, "Axis", "Value"

def get_z_axis_data():
    """
    Extracts the z-axis data from the feature matrix.
    Plots features at indices 12, 15, 18, ...
    """
    X = feature_matrix.X
    num_features = X.shape[0]
    indices = np.arange(12, num_features, 3)
    indices = indices[:4]
    print(f"indices: {indices}")
    data = [X[i] for i in indices]
    data = [d[~np.isnan(d)] for d in data]
    xticklabels = ['B','C','E','F']
    return data, "Z-axis Data", xticklabels, "Axis", "Value"


if __name__ == '__main__':
    # Example: Plotting angle between strands
    data, title, xticklabels, xlabel, ylabel = get_angle_between_strands_data()
   
    plot_violin(data, title, xticklabels, xlabel, ylabel)
    #get_distances_data()
    # To plot other features, uncomment the following lines:
    data, title, xticklabels, xlabel, ylabel = get_projections_data()
    plot_violin(data, title, xticklabels, xlabel, ylabel)

    data, title, xticklabels, xlabel, ylabel = get_distances_data()
    plot_violin(data, title, xticklabels, xlabel, ylabel)
    
    data, title, xticklabels, xlabel, ylabel = get_x_axis_data()
    plot_violin(data, title, xticklabels, xlabel, ylabel)

    data, title, xticklabels, xlabel, ylabel = get_y_axis_data()
    plot_violin(data, title, xticklabels, xlabel, ylabel)

    data, title, xticklabels, xlabel, ylabel = get_z_axis_data()
    plot_violin(data, title, xticklabels, xlabel, ylabel)
