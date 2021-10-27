import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tqdm import tqdm

from notmad.helpers.tf_utils import NOTEARS_loss, DAG_loss
from notmad.helpers.graph_utils import project_to_dag, trim_params

from notears import NOTEARS


class ClusteredNOTEARS:
    def __init__(self, n_clusters, loss_params, context_shape, W_shape,
                 learning_rate=1e-3, clusterer=None, clusterer_fitted=False,
                 tf_dtype=tf.dtypes.float32):
        super(ClusteredNOTEARS, self).__init__()
        if clusterer is None:
            self.clusterer = KMeans(n_clusters=n_clusters)
        else:
            self.clusterer = clusterer # Must have a predict() function
        self.clusterer_fitted = clusterer_fitted
        self.notears_models = [NOTEARS(loss_params, context_shape, W_shape, learning_rate) for i in range(n_clusters)]

    def fit(self, C, X, epochs, batch_size, es_patience=None, val_split=0.25, callbacks=[], verbose=1):
        if len(C.shape) > 2:
            C = C.squeeze()
        if not self.clusterer_fitted:
            k = C.shape[1] // 2
            self.clusterer.fit(C[:,:k]) # TODO: REMOVE CONTEXT SLICING FROM FIT/PREDICT AFTER NEURIPS
            self.clusterer_fitted = True
        train_labels = self.clusterer.predict(C[:,:k])  # TODO: HERE
        loop = list(set(train_labels))
        if verbose:
            loop = tqdm(loop, desc='Clustered NOTEARS Training')
        for clust in loop:
            ct_idxs = train_labels == clust
            if np.sum(ct_idxs) < 2:
                self.notears_models[clust].set_w(np.zeros_like(self.notears_models[clust].get_w()))
            else:
                C_ct = C[ct_idxs]
                X_ct = X[ct_idxs]
                self.notears_models[clust].fit(C_ct, X_ct, epochs, batch_size, es_patience=es_patience, val_split=val_split, callbacks=callbacks, verbose=0)
                self.notears_models[clust].set_w(project_to_dag(self.notears_models[clust].get_w())[0])
            
    def predict_w(self, C, project_to_dag=False):
        # Already projected to DAG space, nothing to do here.
        if len(C.shape) > 2:
            C = C.squeeze()
        k = C.shape[1] // 2
        test_labels  = self.clusterer.predict(C[:,:k])  # TODO: HERE
        return np.array([self.notears_models[label].get_w() for label in test_labels])
    
    def get_ws(self, project_to_dag=False):
        # Already projected to DAG space, nothing to do here.
        return np.array([model.get_w() for model in self.notears_models])
