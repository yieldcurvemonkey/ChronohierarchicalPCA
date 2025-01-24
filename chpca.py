import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

from typing import Dict, List, Tuple, Optional


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return np.dot(u, v)


def _compute_window_hpc(
    t: int,
    window_size: int,
    df: pd.DataFrame,
    preselected_clusters: Dict[str, List[str]],
    asset_list_: List[str],
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Helper function to be run in parallel.
    Computes the HPC eigen-decomposition for window ending at time t, ignoring sign alignment.

    Parameters
    ----------
    t : int
        Current time index in df.
    window_size : int
        Rolling window length.
    df : pd.DataFrame
        Original data: rows=time, cols=assets.
    preselected_clusters : Dict[str, List[str]]
        Mapping of cluster_name -> list_of_assets.
    asset_list_ : List[str]
        Ordered list of all assets.

    Returns
    -------
    (t, w_H, v_H_T) : tuple
        t: integer index,
        w_H: top n_components eigenvalues,
        v_H_T: top n_components eigenvectors (shape: (n_components, n_assets)),
               *without* sign alignment or reordering.
    """
    start_idx = t - window_size + 1
    window_df = df.iloc[start_idx : t + 1]

    # (a) For each cluster, compute 1st eigenvector => cluster factor, then betas
    cluster_benchmarks = {}
    cluster_betas = {}
    for cluster_name, members in preselected_clusters.items():
        subdf = window_df[members]
        cov_ = subdf.cov().values

        w, v = np.linalg.eig(cov_)
        w, v = w.real, v.real

        idx = np.argsort(w)[::-1]
        w, v = w[idx], v[:, idx]

        top_vec = v[:, 0]
        top_eigval = w[0]

        factor_values = (1.0 / np.sqrt(top_eigval)) * subdf.values.dot(top_vec)
        cluster_benchmarks[cluster_name] = factor_values

        # Regress assets on cluster factor => betas
        betas_dict = {}
        F_reshaped = factor_values.reshape(-1, 1)
        for asset in members:
            y = window_df[asset].values
            reg = LinearRegression(fit_intercept=False).fit(F_reshaped, y)
            betas_dict[asset] = reg.coef_[0]
        cluster_betas[cluster_name] = betas_dict

    # (b) Build HPC (hierarchical) matrix
    n_assets = len(asset_list_)
    HPC_matrix = np.zeros((n_assets, n_assets))
    full_cov = window_df.cov().values

    # map each asset -> cluster
    asset_to_cluster = {}
    for c_name, c_members in preselected_clusters.items():
        for a in c_members:
            asset_to_cluster[a] = c_name

    for i, a_i in enumerate(asset_list_):
        c_i = asset_to_cluster[a_i]
        for j, a_j in enumerate(asset_list_):
            c_j = asset_to_cluster[a_j]
            if c_i == c_j:
                HPC_matrix[i, j] = full_cov[i, j]
            else:
                beta_i = cluster_betas[c_i][a_i]
                beta_j = cluster_betas[c_j][a_j]
                F_i = cluster_benchmarks[c_i]
                F_j = cluster_benchmarks[c_j]
                rho = np.corrcoef(F_i, F_j)[0, 1]
                HPC_matrix[i, j] = beta_i * beta_j * rho

    # (c) Eigen-decomposition of HPC_matrix
    w_H, v_H = np.linalg.eigh(HPC_matrix)
    w_H, v_H = w_H.real, v_H.real
    idx_sort = np.argsort(w_H)[::-1]
    w_H = w_H[idx_sort]
    v_H = v_H[:, idx_sort]

    # we slice and normalize later outside
    return t, w_H, v_H


class CHPCA:

    def __init__(self, preselected_clusters: Dict[str, List[str]], window_size: int, n_components: int):
        self.preselected_clusters = preselected_clusters
        self.window_size = window_size
        self.n_components = n_components

        self.components_ = {}  # Dict of t -> (n_components, n_assets) HPC loadings
        self.eigenvals_ = {}  # Dict of t -> (n_components,) HPC eigenvalues
        self.asset_list_ = None  # The list of columns from the input DataFrame
        self.dates_ = None  # The date/time index from the input DataFrame

    def fit(self, df: pd.DataFrame, n_jobs: Optional[int] = None):
        self.asset_list_ = df.columns.tolist()
        self.dates_ = df.index
        T = len(df)

        if n_jobs:
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_window_hpc)(t, self.window_size, df, self.preselected_clusters, self.asset_list_)
                for t in tqdm.tqdm(range(self.window_size - 1, T), desc="ROLLING...")
            )
            raw_results = {}
            for t_out, wH_out, vH_out in parallel_results:
                raw_results[t_out] = (wH_out, vH_out)

            prev_components = None
            for t in sorted(raw_results.keys()):
                w_H, v_H = raw_results[t]
                w_H = w_H[: self.n_components]
                v_H = v_H[:, : self.n_components]  # shape (n_assets, n_components)

                for c_idx in range(self.n_components):
                    norm_ = np.linalg.norm(v_H[:, c_idx])
                    if norm_ > 1e-12:
                        v_H[:, c_idx] /= norm_

                v_H_T = v_H.T
                if prev_components is not None:
                    new_v = np.zeros_like(v_H_T)
                    new_w = np.zeros_like(w_H)
                    used_prev = set()

                    for i_c in range(self.n_components):
                        sims = [abs(_cosine_similarity(v_H_T[i_c], prev_components[j_c])) for j_c in range(self.n_components)]
                        best_j = np.argmax(sims)
                        sorted_idx = np.argsort(sims)[::-1]
                        for candidate_j in sorted_idx:
                            if candidate_j not in used_prev:
                                best_j = candidate_j
                                break
                        used_prev.add(best_j)

                        dot_ = np.dot(v_H_T[i_c], prev_components[best_j])
                        if dot_ < 0:
                            v_H_T[i_c] = -v_H_T[i_c]

                        new_v[best_j] = v_H_T[i_c]
                        new_w[best_j] = w_H[i_c]

                    v_H_T = new_v
                    w_H = new_w

                self.components_[t] = v_H_T  # shape (n_components, n_assets)
                self.eigenvals_[t] = w_H
                prev_components = v_H_T
        else:
            prev_components = None  # used to align HPCs across windows
            for t in tqdm.tqdm(range(self.window_size - 1, T), desc="ROLLING..."):
                start_idx = t - self.window_size + 1
                window_df = df.iloc[start_idx : t + 1]

                # 1) Build the Hierarchical Correlation Matrix for the current window
                #    HPC steps:
                #    a) For each cluster, do PCA => 1st factor (benchmark).
                #    b) Regress each asset in that cluster onto the factor => betas.
                #    c) For cross-cluster blocks, fill with betas * correlation_of_factors.
                #
                #    We do an eigen-decomposition on that HPC matrix.

                # (a) get 1st eigenvector (benchmark) for each cluster
                cluster_benchmarks = {}
                cluster_betas = {}
                for cluster_name, members in self.preselected_clusters.items():
                    subdf = window_df[members]
                    cov_ = subdf.cov().values

                    w, v = np.linalg.eig(cov_)
                    w = w.real
                    v = v.real

                    idx = np.argsort(w)[::-1]
                    w = w[idx]
                    v = v[:, idx]

                    top_vec = v[:, 0]
                    top_eigval = w[0]

                    # build cluster factor timeseries: F^k_t = (1 / sqrt(lambda1)) * sum_i (v_i * X_i)
                    # here subdf is shape (time, #members)
                    # top_vec is (#members,)
                    # So for each row, factor = (1 / sqrt(top_eigval)) * (row . top_vec).
                    # .dot(...) along columns => shape (time,)
                    factor_values = (1.0 / np.sqrt(top_eigval)) * subdf.values.dot(top_vec)
                    cluster_benchmarks[cluster_name] = factor_values

                    # (b) Regress each asset in that cluster on the cluster factor => betas
                    betas_dict = {}
                    F_reshaped = factor_values.reshape(-1, 1)
                    for i, asset in enumerate(members):
                        y = window_df[asset].values
                        reg = LinearRegression(fit_intercept=False).fit(F_reshaped, y)
                        betas_dict[asset] = reg.coef_[0]
                    cluster_betas[cluster_name] = betas_dict

                n_assets = len(self.asset_list_)
                HPC_matrix = np.zeros((n_assets, n_assets))

                full_cov = window_df.cov().values
                asset_to_cluster = {}
                for c_name, c_members in self.preselected_clusters.items():
                    for a in c_members:
                        asset_to_cluster[a] = c_name

                for i, a_i in enumerate(self.asset_list_):
                    cluster_i = asset_to_cluster[a_i]
                    for j, a_j in enumerate(self.asset_list_):
                        cluster_j = asset_to_cluster[a_j]
                        if cluster_i == cluster_j:
                            # same cluster => use empirical cov
                            HPC_matrix[i, j] = full_cov[i, j]
                        else:
                            # cross cluster => sum of (beta_i * beta_j * Corr(F_cluster_i, F_cluster_j))
                            beta_i = cluster_betas[cluster_i][a_i]
                            beta_j = cluster_betas[cluster_j][a_j]

                            F_i = cluster_benchmarks[cluster_i]
                            F_j = cluster_benchmarks[cluster_j]
                            rho = np.corrcoef(F_i, F_j)[0, 1]

                            HPC_matrix[i, j] = beta_i * beta_j * rho

                # 2) Eigen-decompose HPC_matrix
                w_H, v_H = np.linalg.eigh(HPC_matrix)
                w_H = w_H.real
                v_H = v_H.real

                idx_sort = np.argsort(w_H)[::-1]
                w_H = w_H[idx_sort]
                v_H = v_H[:, idx_sort]

                w_H = w_H[: self.n_components]
                v_H = v_H[:, : self.n_components]  # shape: (n_assets, n_components)

                for c_idx in range(self.n_components):
                    norm_ = np.linalg.norm(v_H[:, c_idx])
                    if norm_ > 1e-12:
                        v_H[:, c_idx] /= norm_

                # 3) Align sign & reorder with previous time step (R2-PCA style)
                v_H_T = v_H.T
                if prev_components is not None:
                    used_prev = set()
                    new_v = np.zeros_like(v_H_T)
                    new_w = np.zeros_like(w_H)

                    for i_c in range(self.n_components):
                        # find best matching j_c in prev_components and measure cos sim in absolute value
                        sims = [abs(_cosine_similarity(v_H_T[i_c], prev_components[j_c])) for j_c in range(self.n_components)]
                        best_j = np.argmax(sims)

                        # handle if best_j is already used, pick next
                        sorted_idx = np.argsort(sims)[::-1]
                        for candidate_j in sorted_idx:
                            if candidate_j not in used_prev:
                                best_j = candidate_j
                                break
                        used_prev.add(best_j)

                        # sign alignment
                        dot_ = np.dot(v_H_T[i_c], prev_components[best_j])
                        if dot_ < 0:
                            v_H_T[i_c] = -v_H_T[i_c]

                        new_v[best_j] = v_H_T[i_c]
                        new_w[best_j] = w_H[i_c]

                    v_H_T = new_v
                    w_H = new_w

                self.components_[t] = v_H_T  # shape (n_components, n_assets)
                self.eigenvals_[t] = w_H
                prev_components = v_H_T

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.components_ is None:
            raise ValueError("Must call fit() before transform().")

        T = len(df)
        scores_dict = {}
        for t in range(self.window_size - 1, T):
            row_data = df.iloc[t].values  # shape (n_assets,)
            V_t = self.components_[t]  # shape (n_components, n_assets)
            scores_t = V_t.dot(row_data)
            scores_dict[df.index[t]] = scores_t

        out = pd.DataFrame.from_dict(scores_dict, orient="index")
        out.columns = [f"HPC_{i+1}" for i in range(self.n_components)]
        return out

    def inverse_transform(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        out_data = []
        idx_dates = scores_df.index
        for date_i in idx_dates:
            t = self.dates_.get_loc(date_i)
            if t not in self.components_:
                out_data.append([np.nan] * len(self.asset_list_))
            else:
                V_t = self.components_[t]  # shape (n_components, n_assets)
                sc = scores_df.loc[date_i].values  # shape (n_components,)
                x_hat = sc @ V_t
                out_data.append(x_hat)

        reconstructed_df = pd.DataFrame(data=out_data, index=idx_dates, columns=self.asset_list_)
        return reconstructed_df

    def get_components_at(self, date_idx: int) -> pd.DataFrame:
        """
        Retrieve HPC loadings (eigenvectors) for a specific date index.
        Returns a DataFrame: rows = HPC_i, columns = assets
        """
        if date_idx not in self.components_:
            raise ValueError(f"No HPC stored for t={date_idx}.")

        V_t = self.components_[date_idx]  # (n_components, n_assets)
        df_out = pd.DataFrame(V_t, index=[f"HPC_{i+1}" for i in range(self.n_components)], columns=self.asset_list_)
        return df_out

    def get_eigenvals_at(self, date_idx: int) -> np.ndarray:
        """Return the hierarchical eigenvalues at date_idx."""
        return self.eigenvals_[date_idx]

    def build_pc_loadings(self, plot=False, date_idx=None):
        if self.components_ is None:
            raise ValueError("CHPCA must be fitted before plotting loadings.")

        if not date_idx:
            date_idx = max(self.components_.keys())

        chpca_loadings = {}
        for i in range(self.n_components):
            if plot:
                plt.subplots(tight_layout=True)
                # corr_pcs = np.corrcoef(self.pca_eigenvecs[:, i], self.hpca_eigenvecs[:, i])[0, 1]
                # if corr_pcs < 0:
                #     self.hpca_eigenvecs[:, i] = -self.hpca_eigenvecs[:, i]

                # plt.plot(self.pca_eigenvecs[:, i], label="PCA")
                plt.plot(self.components_[date_idx][i], label="HPCA")
                plt.title("Eigenvector {} -- ".format(i + 1))

                plt.xticks(np.arange(len(self.asset_list_)), self.asset_list_, rotation=90)
                plt.legend()
                plt.show()

            chpca_loadings[f"HPC{i+1}"] = dict(zip(self.asset_list_, self.components_[date_idx][i]))
            # pca_loadings[f"PC{i+1}"] = dict(zip(self.asset_list_, self.pca_eigenvecs[:, i]))

        return chpca_loadings

    def plot_variance_explained(self, date_idx=None):
        if self.components_ is None:
            raise ValueError("CHPCA must be fitted before plotting loadings.")

        if not date_idx:
            date_idx = max(self.components_.keys())

        plt.figure()
        plt.plot((np.cumsum(self.eigenvals_[date_idx]) / self.eigenvals_[date_idx].sum()), label="CHPCA")
        plt.axhline(1, color="k")
        plt.xlabel("Rank k of eigenvalue")
        plt.ylabel("Variance Explained")
        plt.title("Variance explained = sum of the k largest eigenvalues / sum of all eigenvalues")
        plt.legend()
        plt.show()
