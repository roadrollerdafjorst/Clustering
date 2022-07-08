from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster

# Import the data
data = pd.read_csv('Iris.csv')
datadata = data.drop(['Species'], axis=1)
#===========================================================================================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Data Standardization
data_standardized = StandardScaler().fit_transform(datadata)

# Principal component analysis
def pca_func(n):
    pca = PCA(n_components=n)
    pca_data = pca.fit_transform(datadata)
    return pca_data

eig_val, eig_vec = np.linalg.eig(np.cov(pca_func(4).T))

print('Eigenvalues:', eig_val)

# Plot Components v/s Eigenvalues
plt.figure()
plt.bar(['1', '2', '3', '4'], eig_val)
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
plt.grid()
plt.savefig('1.png')

#===========================================================================================================================================
from sklearn.cluster import KMeans

# Data reduced to 2 dimensions
pca_2 = pca_func(2)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pca_2)
kmeans_prediction = kmeans.predict(pca_2)

# Plot scatterplot for reduced data (KMeans Clustering)
plt.figure()
colors = ['r', 'g', 'b']
clusters = np.unique(kmeans_prediction)

for i in clusters:
    plt.scatter(pca_2[kmeans_prediction==i][:, 0], pca_2[kmeans_prediction==i][:, 1], color=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_.T[0], kmeans.cluster_centers_.T[1], marker='*', s=50, color='black', label='Cluster centers')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.savefig('2.png')

print('(b) Distortion measure (K=3):', kmeans.inertia_)

from sklearn import metrics
from scipy.optimize import linear_sum_assignment

# Purity scores function
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    # print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

print('(c) Purity score:', round(purity_score(data.Species, kmeans_prediction), 3))

#===========================================================================================================================================
K = [2, 3, 4, 5, 6, 7]

distortion_measure = []
purity_score_km = []

for i in K:
    km = KMeans(n_clusters=i)
    km.fit(pca_2)
    km_prediction = km.predict(pca_2)

    # Distortion measure for K=i
    dm = km.inertia_
    distortion_measure.append(dm)

    # Purity score for k=i
    ps = round(purity_score(data.Species, km_prediction), 3)
    purity_score_km.append(ps)

# Barplot for K vs distortion measure
plt.figure()
plt.bar(list(map(str, K)), distortion_measure)

plt.xlabel('K')
plt.ylabel('Distortion measure')
plt.grid()
plt.savefig('3a.png')

# Optimal value of K using elbow method
plt.figure()
plt.plot(K, distortion_measure, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion measure')
plt.title('The Elbow Method using Distortion measure')
plt.savefig('3b.png')

optimal_k_km = 5
print('The optimal value of K using elbow method:', optimal_k_km)

print('Purity score for K = [2, 3, 4, 5, 6, 7]:', purity_score_km)

#===========================================================================================================================================
from sklearn.mixture import GaussianMixture

gauss_mix = GaussianMixture(n_components = 3)
gauss_mix.fit(pca_2)
gauss_mix_prediction = gauss_mix.predict(pca_2)

# Plot scatterplot for reduced data (GMM Clustering)
plt.figure()
colors = ['r', 'g', 'b']
clusters = np.unique(gauss_mix_prediction)

for i in clusters:
    plt.scatter(pca_2[gauss_mix_prediction==i][:, 0], pca_2[gauss_mix_prediction==i][:, 1], color=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_.T[0], kmeans.cluster_centers_.T[1], marker='*', s=50, color='black', label='Cluster centers')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.savefig('4.png')

# Total data log likelihood 
print('(b) Total data log likelihood (3 clusters):', round(sum(gauss_mix.score_samples(pca_2)), 3))

# Purity score (K=3)
print('(c) Purity score for K = 3:', round(purity_score(data.Species, gauss_mix_prediction), 3))

#===========================================================================================================================================
total_data_log_likelihood = []
purity_score_gmm = []

for i in K:
    gmm = GaussianMixture(n_components = i)
    gmm.fit(pca_2)
    gmm_prediction = gmm.predict(pca_2)

    # Total data log likelihood for K=i
    tdll = round(sum(gmm.score_samples(pca_2)), 3)
    total_data_log_likelihood.append(tdll)

    # Purity score for k=i
    ps = round(purity_score(data.Species, gmm_prediction), 3)
    purity_score_gmm.append(ps)

print(f'(a) Total data log likelihood for k = {K}:', total_data_log_likelihood)

# Barplot for K vs Total data log likelihood
plt.figure()
plt.bar(list(map(str, K)), total_data_log_likelihood)

plt.xlabel('K')
plt.ylabel('Total data log likelihood')
plt.grid()
plt.savefig('5a.png')

# Optimal value of K using elbow method
plt.figure()
plt.plot(K, total_data_log_likelihood, 'bx-')
plt.xlabel('K')
plt.ylabel('Total data log likelihood')
plt.title('The Elbow Method using Total data log likelihood')
plt.savefig('5b.png')

optimal_k_gmm = 4
print('(b) The optimal value of K using elbow method:', optimal_k_gmm)

print('(c) Purity score for K = [2, 3, 4, 5, 6, 7]:', purity_score_gmm)

#===========================================================================================================================================
eps = [1, 5] # Epsilon
min_samples = [4, 10] # Minimum number of examples present inside the boundary

# DBSCAN
from sklearn.cluster import DBSCAN

fig_i = 1
purity_score_dbs = []

for i in eps:
    for j in min_samples:

        dbscan_model=DBSCAN(eps=i, min_samples=j).fit(pca_2)
        DBSCAN_predictions = dbscan_model.labels_
        
        # Plot scatterplot for reduced data (GMM Clustering)
        plt.figure()
        clusters = np.unique(DBSCAN_predictions)

        for k in clusters:
            plt.scatter(pca_2[DBSCAN_predictions==k][:, 0], pca_2[DBSCAN_predictions==k][:, 1], label=f'Cluster {k+1}')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Epsilon={i} and Minimum number of samples = {j}')
        plt.legend()
        plt.grid()
        plt.savefig(f'6_{fig_i}.png')
        fig_i+=1

        # Purity score for DBSCAN
        psd = round(purity_score(data.Species, DBSCAN_predictions), 3)
        purity_score_dbs.append(psd)

        print(f'Purity score for eps={i} and min_samples={j}:', psd)