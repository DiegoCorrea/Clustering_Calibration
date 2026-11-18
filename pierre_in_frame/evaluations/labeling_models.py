"""
Labeling models used to describe the structure of the users' distributions.

Both the hyperparameter search (step 2) and the stage-wise structural evaluation (step 5) build
the models through this module, so that the selected configuration and the evaluated
configuration are the same object.

Protocol (Section "Stage-wise Structural Evaluation Protocol" of the paper):
- each model is fitted once on the preference distribution (PREF), the reference structure;
- the users of every later stage (candidate items, calibrated lists) are projected onto the
  reference groups, so that all stages share the same group identities.
"""
import time
import warnings

import numpy as np
from fcmeans import FCM
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS, BisectingKMeans, SpectralClustering
)
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from settings.labels import Label

RANDOM_STATE = 42

# Label used by DBSCAN/OPTICS for noise and by the outlier detectors for atypical users.
NOISE_LABEL = -1
OUTLIER_LABEL = -1
INLIER_LABEL = 1

# All outlier detectors flag the same share of users on the preference distribution,
# so that their typical/atypical partitions are comparable.
OUTLIER_CONTAMINATION = 0.1

# Model families.
PARTITION_FAMILY = "PARTITION"
MIXTURE_FAMILY = "MIXTURE"
DENSITY_FAMILY = "DENSITY"
DETECTOR_FAMILY = "DETECTOR"

FAMILY_BY_MODEL = {
    Label.KMEANS: PARTITION_FAMILY,
    Label.BISECTING: PARTITION_FAMILY,
    Label.AGGLOMERATIVE: PARTITION_FAMILY,
    Label.SPECTRAL: PARTITION_FAMILY,
    Label.BIRCH: PARTITION_FAMILY,
    Label.FCM: PARTITION_FAMILY,
    Label.GAUSSIAN_MIXTURE: MIXTURE_FAMILY,
    Label.BAYESIAN_MIXTURE: MIXTURE_FAMILY,
    Label.DBSCAN: DENSITY_FAMILY,
    Label.OPTICS: DENSITY_FAMILY,
    Label.IF: DETECTOR_FAMILY,
    Label.LOF: DETECTOR_FAMILY,
    Label.OSVM: DETECTOR_FAMILY,
    Label.SGD: DETECTOR_FAMILY,
    Label.ENVELOPE: DETECTOR_FAMILY,
}

# Models without out-of-sample prediction. They are projected with a nearest-centroid rule
# (AGGLO, SPECTRAL) or with their own density-reachability rule (DBSCAN, OPTICS).
TRANSDUCTIVE_MODELS = [Label.AGGLOMERATIVE, Label.SPECTRAL, Label.DBSCAN, Label.OPTICS]

# ------------------------------------------------------------------------------------------- #
# Search space
# ------------------------------------------------------------------------------------------- #
# Number of groups/components: every value is evaluated (no random sampling). The grid is
# denser for small values, where the silhouette changes the most.
N_GROUPS_GRID = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

# Density-based models. Duplicated metric names of the first version
# (cityblock = manhattan = l1, euclidean = l2) were removed, as well as "hamming",
# which is not meaningful for continuous distributions.
DENSITY_METRICS = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'braycurtis', 'canberra', 'correlation']
DENSITY_EPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
DENSITY_MIN_SAMPLES = [2, 3, 5, 10, 15, 20, 30, 50, 75, 100]

# Maximum share of noise users accepted for a density-based configuration.
MAX_NOISE_SHARE = 0.5


def family_of(model_name: str) -> str:
    return FAMILY_BY_MODEL[model_name]


def parameter_grid(model_name: str) -> list:
    """
    It returns the full list of configurations of a model.
    """
    family = family_of(model_name)
    if family == PARTITION_FAMILY:
        return [{'n_clusters': k} for k in N_GROUPS_GRID]
    if family == MIXTURE_FAMILY:
        return [{'n_components': k} for k in N_GROUPS_GRID]
    if model_name == Label.DBSCAN:
        return [
            {'eps': eps, 'min_samples': ms, 'metric': metric}
            for eps in DENSITY_EPS for ms in DENSITY_MIN_SAMPLES for metric in DENSITY_METRICS
        ]
    if model_name == Label.OPTICS:
        # OPTICS uses the xi extraction, so eps is not a parameter.
        return [
            {'min_samples': ms, 'metric': metric}
            for ms in DENSITY_MIN_SAMPLES for metric in DENSITY_METRICS
        ]
    # Outlier detectors: one fixed configuration, the same contamination for all of them.
    if model_name == Label.IF:
        return [{'n_estimators': 100, 'contamination': OUTLIER_CONTAMINATION}]
    if model_name == Label.LOF:
        return [{'n_neighbors': 20, 'metric': 'euclidean', 'contamination': OUTLIER_CONTAMINATION}]
    if model_name in [Label.OSVM, Label.SGD]:
        return [{'nu': OUTLIER_CONTAMINATION}]
    if model_name == Label.ENVELOPE:
        return [{'contamination': OUTLIER_CONTAMINATION}]
    raise NameError(f"Labeling model not registered: {model_name}")


def build_model(model_name: str, params: dict):
    """
    It creates a non-fitted instance of the labeling model.
    """
    if model_name == Label.KMEANS:
        return KMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10, random_state=RANDOM_STATE)
    if model_name == Label.BISECTING:
        return BisectingKMeans(n_clusters=params['n_clusters'], init='k-means++', random_state=RANDOM_STATE)
    if model_name == Label.AGGLOMERATIVE:
        return AgglomerativeClustering(n_clusters=params['n_clusters'])
    if model_name == Label.SPECTRAL:
        return SpectralClustering(n_clusters=params['n_clusters'], random_state=RANDOM_STATE)
    if model_name == Label.BIRCH:
        return Birch(n_clusters=params['n_clusters'])
    if model_name == Label.FCM:
        return FCM(n_clusters=params['n_clusters'], random_state=RANDOM_STATE)
    if model_name == Label.GAUSSIAN_MIXTURE:
        return GaussianMixture(
            n_components=params['n_components'], init_params='k-means++', random_state=RANDOM_STATE
        )
    if model_name == Label.BAYESIAN_MIXTURE:
        return BayesianGaussianMixture(
            n_components=params['n_components'], init_params='k-means++', max_iter=500,
            random_state=RANDOM_STATE
        )
    if model_name == Label.DBSCAN:
        return DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=params['metric'])
    if model_name == Label.OPTICS:
        return OPTICS(min_samples=params['min_samples'], metric=params['metric'])
    if model_name == Label.IF:
        return IsolationForest(
            n_estimators=params['n_estimators'], contamination=params['contamination'], random_state=RANDOM_STATE
        )
    if model_name == Label.LOF:
        # novelty=True enables the projection of the later stages.
        return LocalOutlierFactor(
            n_neighbors=params['n_neighbors'], metric=params['metric'],
            contamination=params['contamination'], novelty=True
        )
    if model_name == Label.OSVM:
        return OneClassSVM(nu=params['nu'])
    if model_name == Label.SGD:
        return SGDOneClassSVM(nu=params['nu'], random_state=RANDOM_STATE)
    if model_name == Label.ENVELOPE:
        return EllipticEnvelope(contamination=params['contamination'], random_state=RANDOM_STATE)
    raise NameError(f"Labeling model not registered: {model_name}")


class ReferenceStructure:
    """
    A labeling model fitted on the preference distribution, which assigns users of any stage
    to the reference groups.
    """

    def __init__(self, model_name: str, params: dict):
        self.model_name = model_name
        self.family = family_of(model_name)
        self.params = params
        self.model = build_model(model_name, params)

        self.fit_labels_ = None
        self.fit_time_ = None

        self._centroids = None
        self._centroid_labels = None
        self._neighbors = None
        self._neighbor_labels = None
        self._reach_radius = None

    def fit(self, x: np.ndarray):
        """
        It fits the model on the preference distribution and keeps the labels found by the model.
        """
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.model_name in TRANSDUCTIVE_MODELS:
                self.fit_labels_ = np.asarray(self.model.fit_predict(x))
                self._prepare_projection(x)
            elif self.model_name == Label.LOF:
                self.model.fit(x)
                # Labels of the training users, equal to LocalOutlierFactor(novelty=False).fit_predict.
                self.fit_labels_ = np.where(
                    self.model.negative_outlier_factor_ < self.model.offset_, OUTLIER_LABEL, INLIER_LABEL
                )
            else:
                self.model.fit(x)
                self.fit_labels_ = np.asarray(self.model.predict(x))
        self.fit_time_ = time.perf_counter() - start
        return self

    def _prepare_projection(self, x: np.ndarray):
        labels = self.fit_labels_
        if self.model_name in [Label.AGGLOMERATIVE, Label.SPECTRAL]:
            self._centroid_labels = np.unique(labels)
            self._centroids = np.vstack([x[labels == lab].mean(axis=0) for lab in self._centroid_labels])
        elif self.model_name == Label.DBSCAN:
            # A new user joins the cluster of the nearest core user within eps, otherwise it is noise.
            core = self.model.core_sample_indices_
            if len(core) > 0:
                self._neighbors = NearestNeighbors(n_neighbors=1, metric=self.params['metric']).fit(x[core])
                self._neighbor_labels = labels[core]
                self._reach_radius = np.full(len(core), self.params['eps'])
        elif self.model_name == Label.OPTICS:
            # A new user joins the cluster of the nearest clustered user p if it lies within the
            # core distance of p (density reachability), otherwise it is noise.
            clustered = np.where(labels != NOISE_LABEL)[0]
            if len(clustered) > 0:
                self._neighbors = NearestNeighbors(
                    n_neighbors=min(len(clustered), 50), metric=self.params['metric']
                ).fit(x[clustered])
                self._neighbor_labels = labels[clustered]
                self._reach_radius = self.model.core_distances_[clustered]

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        It assigns the users of a stage to the reference groups.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.model_name in [Label.AGGLOMERATIVE, Label.SPECTRAL]:
                dist = ((x[:, None, :] - self._centroids[None, :, :]) ** 2).sum(axis=2)
                return self._centroid_labels[np.argmin(dist, axis=1)]
            if self.model_name in [Label.DBSCAN, Label.OPTICS]:
                if self._neighbors is None:
                    return np.full(x.shape[0], NOISE_LABEL)
                dist, idx = self._neighbors.kneighbors(x)
                reachable = dist <= self._reach_radius[idx]
                labels = np.full(x.shape[0], NOISE_LABEL)
                has_reach = reachable.any(axis=1)
                first = np.argmax(reachable, axis=1)
                labels[has_reach] = self._neighbor_labels[idx[has_reach, first[has_reach]]]
                return labels
            return np.asarray(self.model.predict(x))


# ------------------------------------------------------------------------------------------- #
# Structure descriptors
# ------------------------------------------------------------------------------------------- #
def excluded_mask(model_name: str, labels: np.ndarray) -> np.ndarray:
    """
    Users that do not belong to any group (noise of the density-based models).
    The atypical users of the outlier detectors form a group (the typical/atypical partition).
    """
    if family_of(model_name) == DENSITY_FAMILY:
        return labels == NOISE_LABEL
    return np.zeros(len(labels), dtype=bool)


def n_groups(model_name: str, labels: np.ndarray) -> int:
    return int(len(np.unique(labels[~excluded_mask(model_name, labels)])))


def silhouette(model_name: str, x: np.ndarray, labels: np.ndarray) -> float:
    """
    Signed silhouette coefficient with the Euclidean distance, computed on the grouped users.
    It is undefined (NaN) when fewer than two groups are occupied.
    """
    keep = ~excluded_mask(model_name, labels)
    k = len(np.unique(labels[keep]))
    if k < 2 or k > keep.sum() - 1:
        return np.nan
    return float(silhouette_score(x[keep], labels[keep], metric='euclidean'))


def group_size_descriptors(model_name: str, labels: np.ndarray) -> dict:
    """
    Number of groups, effective number of groups (exponential of the entropy of the group sizes),
    share of the largest group and share of excluded users (noise).
    """
    excluded = excluded_mask(model_name, labels)
    grouped = labels[~excluded]
    if len(grouped) == 0:
        return {'n_groups': 0, 'effective_groups': 0.0, 'largest_group_share': np.nan,
                'excluded_share': float(excluded.mean())}
    _, counts = np.unique(grouped, return_counts=True)
    share = counts / counts.sum()
    return {
        'n_groups': int(len(counts)),
        'effective_groups': float(np.exp(-(share * np.log(share)).sum())),
        'largest_group_share': float(share.max()),
        'excluded_share': float(excluded.mean()),
    }


def outlier_share(model_name: str, labels: np.ndarray) -> float:
    if family_of(model_name) == DETECTOR_FAMILY:
        return float((labels == OUTLIER_LABEL).mean())
    if family_of(model_name) == DENSITY_FAMILY:
        return float((labels == NOISE_LABEL).mean())
    return 0.0
