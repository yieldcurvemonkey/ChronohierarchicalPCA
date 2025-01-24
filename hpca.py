from typing import Dict, List, Optional

import fastcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.cluster import hierarchy



def compute_sorted_correlations(timeseries_df: pd.DataFrame, corr_estimator="pearson"):
    corr = timeseries_df.corr(method=corr_estimator)
    dist = 1 - corr.values
    tri_a, tri_b = np.triu_indices(len(dist), k=1)
    linkage = fastcluster.linkage(dist[tri_a, tri_b], method="ward")
    permutation = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, dist[tri_a, tri_b]))
    sorted_permuted_vols = timeseries_df.columns[permutation]
    return pd.DataFrame(corr.values[permutation, :][:, permutation], index=sorted_permuted_vols, columns=sorted_permuted_vols)


class HPCA:
    def __init__(self, n_components: int, preselected_clusters: Optional[Dict[str, List[str]]] = None, n_clusters: Optional[int] = 5):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.preselected_clusters = preselected_clusters or {}
        self.eigen_clusters = {}
        self.ts_to_cluster = {}
        self.HPCA_cov = None
        self.hpca_eigenvals = None
        self.hpca_eigenvecs = None
        self.pca_eigenvals = None
        self.pca_eigenvecs = None
        self.betas = {}
        self.transformed_scores = None
        self.corr_cols = None
        self.corr_idx = None

    def fit(self, returns_df: pd.DataFrame, n_eigenportfolios: Optional[int] = 1, show_cluster_plots: Optional[bool] = False):
        if self.preselected_clusters is None:
            sorted_correlations = compute_sorted_correlations(returns_df)
            dist = 1 - sorted_correlations.values
            dim = len(dist)
            tri_a, tri_b = np.triu_indices(dim, k=1)
            linkage = fastcluster.linkage(dist[tri_a, tri_b], method="ward")
            clustering_inds = hierarchy.fcluster(linkage, self.n_clusters, criterion="maxclust")
            clusters = {i: [] for i in range(min(clustering_inds), max(clustering_inds) + 1)}
            for i, v in enumerate(clustering_inds):
                clusters[v].append(i)

            permutation = sorted([(min(elems), c) for c, elems in clusters.items()], key=lambda x: x[0], reverse=False)
            sorted_clusters = {}
            for cluster in clusters:
                sorted_clusters[cluster] = clusters[permutation[cluster - 1][1]]
            
            if show_cluster_plots:
                plt.figure(figsize=(5, 5))
                plt.pcolormesh(sorted_correlations, cmap="coolwarm")
                for _, cluster in sorted_clusters.items():
                    xmin, xmax = min(cluster), max(cluster)
                    ymin, ymax = min(cluster), max(cluster)

                    plt.axvline(x=xmin, ymin=ymin / dim, ymax=(ymax + 1) / dim, color="r")
                    plt.axvline(x=xmax + 1, ymin=ymin / dim, ymax=(ymax + 1) / dim, color="r")
                    plt.axhline(y=ymin, xmin=xmin / dim, xmax=(xmax + 1) / dim, color="r")
                    plt.axhline(y=ymax + 1, xmin=xmin / dim, xmax=(xmax + 1) / dim, color="r")
                plt.show()

            for cluster in sorted_clusters:
                cluster_members = sorted_correlations.columns[sorted_clusters[cluster]].tolist()
                for asset in cluster_members:
                    self.preselected_clusters[asset] = cluster
                returns_df[cluster_members].cumsum().plot(legend=(len(cluster_members) < 30))
                plt.show()

        self._assign_clusters()
        self._compute_eigen_clusters(returns_df, n_eigenportfolios)
        self._compute_betas(returns_df, n_eigenportfolios)
        self._build_hpca_covariance(returns_df, n_eigenportfolios)
        self._compute_hpca_eigenvalues(returns_df)
        self.corr_cols = returns_df.columns
        self.corr_idx = returns_df.index

    def fit_transform(self, returns_df, m=None):
        self.fit(returns_df)
        self.transformed_scores = self.transform(returns_df, m)
        return self.transformed_scores

    def _assign_clusters(self):
        for cluster, cluster_members in self.preselected_clusters.items():
            for col in cluster_members:
                self.ts_to_cluster[col] = cluster

    def _compute_eigen_clusters(self, returns_df: pd.DataFrame, n_eigenportfolios: Optional[int] = 1):
        for cluster, cluster_members in self.preselected_clusters.items():
            cluster_returns = returns_df[cluster_members]
            cov_cluster = cluster_returns.cov()

            eigenvals, eigenvecs = np.linalg.eig(cov_cluster.values)
            eigenvals = eigenvals.real
            eigenvecs = eigenvecs.real

            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            cluster_data = {
                "tickers": cluster_members,
                "eigenvals": eigenvals[:n_eigenportfolios],
                "eigenvecs": eigenvecs[:, :n_eigenportfolios],
            }

            # Compute F_k for each eigenportfolio
            for k in range(n_eigenportfolios):
                val_k = eigenvals[k]
                vec_k = eigenvecs[:, k]
                F_k = (1 / np.sqrt(val_k)) * np.multiply(vec_k, cluster_returns.values).sum(axis=1)
                cluster_data[f"F{k+1}"] = F_k

            self.eigen_clusters[cluster] = cluster_data

    def _compute_betas(self, returns_df: pd.DataFrame, n_eigenportfolios: Optional[int] = 1):
        for col in returns_df.columns:
            cluster = self.ts_to_cluster[col]
            self.betas[col] = [
                LinearRegression(fit_intercept=False).fit(self.eigen_clusters[cluster][f"F{k+1}"].reshape(-1, 1), returns_df[col]).coef_[0]
                for k in range(n_eigenportfolios)
            ]

    def _build_hpca_covariance(self, returns_df: pd.DataFrame, n_eigenportfolios: Optional[int] = 1):
        self.HPCA_cov = returns_df.cov()
        for col_1 in self.HPCA_cov.columns:
            for col_2 in self.HPCA_cov.columns:
                if self.ts_to_cluster[col_1] != self.ts_to_cluster[col_2]:
                    mod_rho = 0
                    for k in range(n_eigenportfolios):
                        beta_1 = self.betas[col_1][k]
                        F1_k = self.eigen_clusters[self.ts_to_cluster[col_1]][f"F{k+1}"]
                        beta_2 = self.betas[col_2][k]
                        F2_k = self.eigen_clusters[self.ts_to_cluster[col_2]][f"F{k+1}"]
                        rho_sector = np.corrcoef(F1_k, F2_k)[0, 1]
                        mod_rho += beta_1 * beta_2 * rho_sector
                    self.HPCA_cov.at[col_1, col_2] = mod_rho

    def _compute_hpca_eigenvalues(self, returns_df: pd.DataFrame):
        eigenvals, eigenvecs = np.linalg.eig(self.HPCA_cov.values)
        self.hpca_eigenvals = eigenvals.real
        self.hpca_eigenvecs = eigenvecs.real
        idx = self.hpca_eigenvals.argsort()[::-1]
        self.hpca_eigenvals = self.hpca_eigenvals[idx]
        self.hpca_eigenvecs = self.hpca_eigenvecs[:, idx]

        eigenvals, eigenvecs = np.linalg.eig(returns_df.cov().values)
        self.pca_eigenvals = eigenvals.real
        self.pca_eigenvecs = eigenvecs.real
        idx = self.pca_eigenvals.argsort()[::-1]
        self.pca_eigenvals = eigenvals[idx]
        self.pca_eigenvecs = eigenvecs[:, idx]

    def _calculate_factor_scores(self, returns_df, m=None):
        factor_scores = np.zeros_like(returns_df.values)
        m = m or len(self.hpca_eigenvals)

        for k in range(m):
            score_k = (1 / np.sqrt(self.hpca_eigenvals[k])) * (returns_df.values @ self.hpca_eigenvecs[:, k])
            factor_scores += np.outer(score_k, self.hpca_eigenvecs[:, k])

        return pd.DataFrame(factor_scores, index=returns_df.index, columns=returns_df.columns)

    def transform(self, returns_df, m=None):
        if self.hpca_eigenvals is None or self.hpca_eigenvecs is None:
            raise ValueError("HPCA must be fitted before calling transform.")

        factor_scores = []
        m = m or self.n_components 

        for k in range(m):
            score_k = (1 / np.sqrt(self.hpca_eigenvals[k])) * (returns_df.values @ self.hpca_eigenvecs[:, k])
            factor_scores.append(score_k)

        return np.array(factor_scores).T

    def inverse_transform(self, transformed_scores=None, m=None):
        if self.hpca_eigenvals is None or self.hpca_eigenvecs is None:
            raise ValueError("HPCA must be fitted before calling inverse_transform.")

        if transformed_scores is None:
            if self.transformed_scores is None:
                raise ValueError("No transformed scores available. Please fit_transform or provide transformed_scores.")
            transformed_scores = self.transformed_scores

        reconstructed_returns = np.zeros((transformed_scores.shape[0], self.hpca_eigenvecs.shape[0]))
        m = m or self.n_components 

        for k in range(m):
            reconstructed_returns += np.outer(transformed_scores[:, k], self.hpca_eigenvecs[:, k]) * np.sqrt(self.hpca_eigenvals[k])

        return pd.DataFrame(reconstructed_returns, index=self.corr_idx, columns=self.corr_cols)

    def plot_variance_explained(self, eigvecs_to_plot: Optional[int] = 10):
        if self.hpca_eigenvals is None:
            raise ValueError("HPCA must be fitted before plotting variance explained.")

        plt.figure()
        plt.plot((np.cumsum(self.hpca_eigenvals) / self.hpca_eigenvals.sum())[:eigvecs_to_plot], label="HPCA")
        plt.plot((np.cumsum(self.pca_eigenvals) / self.pca_eigenvals.sum())[:eigvecs_to_plot], label="PCA")
        plt.axhline(1, color="k")
        plt.xlabel("Rank k of eigenvalue")
        plt.ylabel("Variance Explained")
        plt.title("Variance explained = sum of the k largest eigenvalues / sum of all eigenvalues")
        plt.legend()
        plt.show()

    def build_pc_loadings(self, plot=False):
        if self.hpca_eigenvecs is None:
            raise ValueError("HPCA must be fitted before plotting loadings.")

        hpca_loadings = {}
        pca_loadings = {}
        for i in range(self.n_components):
            if plot:
                plt.subplots(tight_layout=True)
                corr_pcs = np.corrcoef(self.pca_eigenvecs[:, i], self.hpca_eigenvecs[:, i])[0, 1]
                if corr_pcs < 0:
                    self.hpca_eigenvecs[:, i] = -self.hpca_eigenvecs[:, i]

                plt.plot(self.pca_eigenvecs[:, i], label="PCA")
                plt.plot(self.hpca_eigenvecs[:, i], label="HPCA")
                plt.title(
                    "Eigenvector {} -- ".format(i + 1) + "Correlation between PC{} and HPC{} = {}".format(i + 1, i + 1, round(abs(corr_pcs), 2)),
                )

                plt.xticks(np.arange(len(self.corr_cols)), self.corr_cols, rotation=90)
                plt.legend()
                plt.show()

            hpca_loadings[f"HPC{i+1}"] = dict(zip(self.corr_cols, self.hpca_eigenvecs[:, i]))
            pca_loadings[f"PC{i+1}"] = dict(zip(self.corr_cols, self.pca_eigenvecs[:, i]))

        return hpca_loadings, pca_loadings
