import os
import sys
import copy
import tensorflow as tf
import tensorflow_addons as tfa
import tqdm
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from notmad.baselines import NOTEARS, ClusteredNOTEARS
from notmad.notmad import NOTMAD
from notmad.helpers import utils
from notmad.helpers import graph_utils
import dataloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_dtype = tf.dtypes.float32
threshs = [0.1, 0.2] #[0.0, 0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def print_header(model_names, out_file):
    print("\t".join(['n','d', 'n_edges', 'd_context', 'context_snr', 'k_true', 'k']), end="\t", file=out_file)
    for i, model_name in enumerate(model_names):
        print("\t".join(["{}_{:.3f}_recovery".format(model_name, x) for x in threshs]), end="\t", file=out_file)
        end = '\n'
        if i < len(model_names) - 1:
            end = '\t'
        print("\t".join(["{}_{:.3f}_mse".format(model_name, x) for x in threshs]), end=end, file=out_file)


def print_results(model_names, data_params, k, results, out_file):
    print("{}\t{}\t{}\t{}\t{:.3f}\t{}\t{}".format(
        data_params["n"], data_params['d'], 
        data_params["n_edges"], data_params["n_c"],
        data_params["context_snr"], data_params["k_true"], 
        k),
          end='\t', file=out_file)
    for i, model_name in enumerate(model_names):
        print("\t".join(["{:.3f}".format(x) for x in results[model_name]['recovery']]),
              end='\t', file=out_file)
        end = '\n'
        if i < len(model_names) - 1:
            end = '\t'
        print("\t".join(["{:.3f}".format(x) for x in results[model_name]['mse']]),
              end=end, flush=True, file=out_file)


def fit_pop(loss_params, C_train, X_train):
    W_shape = (X_train.shape[-1], X_train.shape[-1])
    notears = NOTEARS(loss_params, C_train.shape, W_shape, learning_rate=1e-2) 
    notears.fit(C_train, X_train, epochs=1000, batch_size=32, es_patience=1, verbose=1)
    return notears


def fit_clustered(loss_params, C_train, X_train, k):
    W_shape = (X_train.shape[-1], X_train.shape[-1])
    clustered = ClusteredNOTEARS(k, loss_params, C_train.shape, W_shape,
                 learning_rate=1e-2, tf_dtype=tf.dtypes.float32)
    clustered.fit(C_train, X_train, epochs=1000, batch_size=32, es_patience=1,
                  val_split=0.25, verbose=1)
    return clustered


def fit_notmad(sample_specific_loss_params, archetype_loss_params, 
                  C_train, X_train, k, project, notears_pop, base_predictor):
    init_mat = np.random.uniform(-0.01, 0.01, size=(k, X_train.shape[-1], X_train.shape[-1])) #np.zeros((k, X_train.shape[-1], X_train.shape[-1])) #
#     init_mat = init_mat + notears_pop.get_w()
#     init_mat = np.random.uniform(-0.1, 0.1, size=(k, X_train.shape[-1], X_train.shape[-1]))#np.tile(notears_pop, (k, 1, 1))
#     init_mat = np.array([graph_utils.project_to_dag(mat)[0] for mat in init_mat])
#     init_mat = base_predictor.get_ws()
    make_notmad = lambda: NOTMAD(
        C_train.shape, X_train.shape, k,
        sample_specific_loss_params, archetype_loss_params,
        n_encoder_layers=1, encoder_width=32,
        activation='linear', init_mat=init_mat,
        learning_rate=1e-3, project_archs_to_dag=project, # TODO: should this be variable?
        project_distance=1.0,
        context_activity_regularizer=tf.keras.regularizers.l1(0),
        use_compatibility=False, update_compat_by_grad=False,
        pop_model=None, base_predictor=base_predictor
    )
    notmad = make_notmad()
    notmad.fit(C_train, X_train, batch_size=1, epochs=20, es_patience=2, verbose=1)
    return notmad


def run_experiment(data_params, k, threshs, model_names):
    results = {name: {'f1': [], 'mse': []} for name in model_names}    
    W, C, X, W_dict, C_dict = dataloader.gen_data(data_params)
#     pca = PCA(n_components=3)
#     X_small = pca.fit_transform(X.squeeze())
#     C = np.hstack((C, X_small))
    C_train, C_test, X_train, X_test, W_train, W_test = train_test_split(C, X, W,
                                                                         test_size=0.25)
    
    def calc_recovery_errs(preds, W):
        return [np.linalg.norm(preds[i] - W[i], ord=2) for i in range(len(preds))]
    
    def calc_recovery_err(preds, W):
        return [np.mean(calc_recovery_errs(preds, W)) for thresh in threshs]
    
    def add_results(name, preds):
        results[name]['recovery'] = calc_recovery_err(preds, W_test)
        results[name]['mse'] = [np.mean(utils.mses_xw(X_test, preds*np.abs(preds) > thresh)) for thresh in threshs]

    add_results('base', np.ones_like(W_test))
    # print('Base', results['base']['recovery'])
    
    loss_params = {'l1': 1e-3,
                   'alpha': 1e-2,
                   'rho':1e-1}
    notears = fit_pop(loss_params, C_train, X_train)
    notears_preds = notears.predict_w(C_test, project_to_dag=True)
    add_results('notears', notears_preds)
    # print('Pop', results['notears']['recovery'])
    
    clustered = fit_clustered(loss_params, C_train, X_train, data_params['k_true'])
    cluster_preds = clustered.predict_w(C_test, project_to_dag=True)
    add_results("cluster", cluster_preds)
    # print('Cluster', results['cluster']['recovery'])
    
    sample_specific_loss_params = {'l1': 0., 'alpha': 2e1, 'rho': 1e0}
    archetype_loss_params = {'l1': 0., 'alpha': 1e-1, 'rho': 1e-2}

    notmad = fit_notmad(
        sample_specific_loss_params, archetype_loss_params,
        C_train, X_train, k, project=True, notears_pop=None, base_predictor=clustered)
    preds = notmad.predict_w(C_test, project_to_dag=True).squeeze()
    add_results('notmad', preds)
    # print('NOTMAD', results['notmad']['recovery'])
    
    notmad_nobase = fit_notmad(
        sample_specific_loss_params, archetype_loss_params,
        C_train, X_train, k, project=True, notears_pop=None, base_predictor=None)
    preds_nobase = notmad_nobase.predict_w(C_test, project_to_dag=True).squeeze()
    add_results('notmad_nobase', preds_nobase)
    # print('NOTMAD_nobase', results['notmad_nobase']['recovery'])
    
    train_preds_not_projected = notmad.predict_w(C_train, project_to_dag=False).squeeze()
    train_preds = notmad.predict_w(C_train, project_to_dag=True).squeeze()
    # print("pop train", calc_recovery_err(notears.predict_w(C_train), W_train))
    # print("pop train projected", calc_recovery_err(notears.predict_w(C_train, project_to_dag=True), W_train))
    # print("context train", calc_recovery_err(train_preds_not_projected, W_train))
    # print("context train projected", calc_recovery_err(train_preds, W_train))
    
    # print("pop test", calc_recovery_err(notears.predict_w(C_test), W_test))
    # print("context test", calc_recovery_err(preds, W_test))
    
#     fig = plt.figure()
#     plt.hist(calc_recovery_errs(notears_preds, W_test), label='NOTEARS')
#     plt.hist(calc_recovery_errs(cluster_preds, W_test), label='Cluster')
#     plt.hist(calc_recovery_errs(preds, W_test), label='NOTMAD')
#     plt.legend()
#     plt.show()
    
    """
    fig = plt.figure()
    plt.imshow(notears_preds[0])
    fig = plt.figure()
    plt.imshow(cluster_preds[0])
    fig = plt.figure()
    plt.imshow(preds[0])
    plt.show()
    """

    
    """
    sample_specific_loss_params = {'l1':1e-3, 'alpha': 1e0, 'rho': 1e-1}
    archetype_loss_params = {'l1':0, 'alpha': 0, 'rho': 0}
    lr_notmad = fit_lr_notmad(sample_specific_loss_params, archetype_loss_params,
                              C_train, X_train, k, rank, project=False, notears_pop=notears.get_w())
    preds = lr_notmad.predict_w(C_test, project_to_dag=True).squeeze()
    add_results('lr_notmad', preds)
    """
    
    """
    notmad = fit_notmad(sample_specific_loss_params, archetype_loss_params,
                              C_train, X_train, k, rank, project=True)
    preds = notmad.predict_w(C_test, project_to_dag=True).squeeze()
    add_results('notmad_project', preds)
    """
    
    """
    print("Fitting LIONESS...")
    lioness = LIONESS()
    lioness.fit(loss_params, C_train, X_train, init_model=notears, es_patience=1)
    W_lioness = lioness.Ws
    print("Finished fitting LIONESS.")
    f1s_lioness   = utils.get_f1s(W_train, W_lioness, threshs) # TODO: what's the most fair way to compare?
    """
    return results


if __name__ == "__main__":
    model_names = ['base', 'notears', 'cluster', 'notmad', 'notmad_nobase'] #'lioness'

    # TODO: X Noise scale

    # for graph_type in ["ER", "SF", "BP"]:
    for graph_type in ["ER"]:
        data_params = {
            "n_i": 1,    # number of samples per DAG
            "n_c": 0,    # number of contextual features
            "simulation_type": 'clusters',  # archetypes, clusters, random
            "ensure_convex" : False, # should the archetype be generated such that they form a convex set of DAGs?      
            "graph_type": graph_type,
            'sem_type' : 'gauss',
            "n": 1000,
            "d": 6,
            "n_edges": 6,
            "k_true": 8,
    #         "n_mix": 2,
            "arch_min_radius": 100,
            "cluster_max_radius": 0.2,
            "context_snr": 0.75, 
        }
        k = 8
        
        filepath = "results/simulation_results_{}.tsv".format(graph_type.lower())
        if not os.path.exists(filepath):
            os.makedirs('results', exist_ok=True)
            with open(filepath, 'w') as outfile:
                print_header(model_names, outfile)
        with open(filepath, 'a') as outfile:    
            results = run_experiment(data_params, k, threshs, model_names)
            print_results(model_names, data_params, k, results, outfile)
        print(f'Saved to {filepath}')
        exit()

