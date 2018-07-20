import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns=good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the feature weights as a function of the components
    components.plot(ax=ax, kind='bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i - 0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f" % (ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)

def cluster_results(reduced_data, preds, centers, pca_samples):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and student-selected sample data
    '''

    predictions = pd.DataFrame(preds, columns=['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis=1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map
    cmap = cm.get_cmap('gist_rainbow')

    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2', \
                     color=cmap((i) * 1.0 / (len(centers) - 1)), label='Cluster %i' % (i), s=30);

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x=c[0], y=c[1], color='white', edgecolors='black', \
                   alpha=1, linewidth=2, marker='o', s=200);
        ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100);

    # Plot transformed sample points
    ax.scatter(x=pca_samples[:, 0], y=pca_samples[:, 1], \
               s=150, linewidth=4, color='black', marker='x');

    # Set plot title
    ax.set_title(
        "Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");

def biplot(good_data, reduced_data, pca):
    fig, ax = plt.subplots(figsize=(14, 8))
    # scatterplot of the reduced data
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'],
               facecolors='b', edgecolors='b', s=70, alpha=0.5)

    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1],
                 head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0] * text_pos, v[1] * text_pos, good_data.columns[i], color='black',
                ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax


def channel_results(reduced_data, outliers, pca_samples):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
    Data is labeled by "Channel" and cues added for student-selected sample data
    '''

    # Check that the dataset is loadable
    try:
        full_data = pd.read_csv("customers.csv")
    except:
        print "Dataset could not be loaded. Is the file missing?"
        return False

    # Create the Channel DataFrame
    channel = pd.DataFrame(full_data['Channel'], columns=['Channel'])
    channel = channel.drop(channel.index[outliers]).reset_index(drop=True)
    labeled = pd.concat([reduced_data, channel], axis=1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map
    cmap = cm.get_cmap('gist_rainbow')

    # Color the points based on assigned Channel
    labels = ['Hotel/Restaurant/Cafe', 'Retailer']
    grouped = labeled.groupby('Channel')
    for i, channel in grouped:
        channel.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2', \
                     color=cmap((i - 1) * 1.0 / 2), label=labels[i - 1], s=30);

    # Plot transformed sample points
    for i, sample in enumerate(pca_samples):
        ax.scatter(x=sample[0], y=sample[1], \
                   s=200, linewidth=3, color='black', marker='o', facecolors='none');
        ax.scatter(x=sample[0] + 0.25, y=sample[1] + 0.3, marker='$%d$' % (i), alpha=1, s=125);

    # Set plot title
    ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled");


data = pd.read_csv("customers.csv")
data.drop(['Region', 'Channel'], axis=1, inplace=True)
print data.keys()

indices = [65, 95, 71]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)

log_data = np.log(data)
log_samples = np.log(samples)

outlier_set = set()
for feature in log_data.keys():
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)

    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)

    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5

    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    outlier_f = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    outlier_set.update(outlier_f.index.tolist())

outliers = sorted(list(outlier_set))
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(good_data)
reduced_data = pca.transform(good_data)
pca_samples = pca.transform(log_samples)
reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# Create a biplot
biplot(good_data, reduced_data, pca)

cluster_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
best_score = 0.0
for n in cluster_sizes:
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    from sklearn.mixture import GaussianMixture

    clusterer = GaussianMixture(n_components=n, random_state=42)
    clusterer.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    from sklearn.metrics import silhouette_score

    score = silhouette_score(reduced_data, preds)
    print "clusters of sz={} yields silhouette score : {}".format(n, score)
    if score > best_score:
        result = (clusterer, preds, centers, sample_preds, score, n)
        best_score = score

print "Best Score is {0:.4f} , with {1:2d} clusters".format(best_score, result[5])

preds = result[1]
centers = result[2]
sample_preds = result[3]

cluster_results(reduced_data, preds, centers, pca_samples)

log_centers = pca.inverse_transform(centers)
true_centers = np.exp(log_centers)

segments = ['Segment {}'.format(i) for i in range(0, len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns=data.keys())
true_centers.index = segments

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred

# Display the clustering results based on 'Channel' data
channel_results(reduced_data, outliers, pca_samples)
