import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback

from notmad.helpers.tf_utils import NOTEARS_loss, DAG_loss
from notmad.helpers import graph_utils

# Keras NOTEARS version which takes in context and returns a single population model.

class DummyWeight(tf.keras.layers.Layer):
    def __init__(self, W_shape):
        super(DummyWeight, self).__init__()

        # Define the trainable networks
        self.W = self.add_weight("W", shape=W_shape,
            initializer=tf.keras.initializers.Constant(np.zeros(W_shape)))

    def build(self, input_shapes):
        pass

    def call(self, _):
        return self.W

# TODO: Sync this callback with the callback for CNOTEARS
class DynamicAlphaRho(Callback):
    def __init__(self, C_train):
        super(DynamicAlphaRho, self).__init__()
        self.C_train = C_train
        self.h_old = 0.
        
    def on_epoch_begin(self, epoch, logs=None):
        pred = np.squeeze(self.model.predict(np.expand_dims(
            self.C_train[np.random.choice(self.C_train.shape[0])], 0)))
        #pred = trim_params(pred, thresh=0.1)
        my_dag_loss = DAG_loss(pred, self.model.alpha.numpy(), self.model.rho.numpy()) # TODO: should be measured over batch
        self.model.W.W.assign(self.model.W.W*(1-np.eye(self.model.W.W.shape[0]))) # set the diagonal to 0
        if my_dag_loss > 0.25*self.h_old:
            self.model.alpha.assign(self.model.alpha+self.model.rho*my_dag_loss)
            self.model.rho.assign(self.model.rho*10)
            #self.model.rho.assign(self.model.rho*1.1)
            # print(self.model.alpha.numpy(), self.model.rho.numpy())
        self.h_old = my_dag_loss

        
class NOTEARS:
    def __init__(self, loss_params, context_shape, W_shape,
                 learning_rate=1e-3,
                 tf_dtype=tf.dtypes.float32):
        super(NOTEARS, self).__init__()
        encoder_input_shape = (context_shape[1], 1)
        self.context = tf.keras.layers.Input(
            shape=encoder_input_shape, dtype=tf_dtype, name="C")
        self.W = DummyWeight(W_shape)
        #self.W = tf.Variable(initial_value=np.zeros(W_shape), trainable=True)
        self.outputs = self.W(self.context)
        self.model = tf.keras.models.Model(inputs=self.context, outputs=self.outputs)

        # Compile the model
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = []
        try:
            self.model.alpha = tf.Variable(loss_params['init_alpha'], trainable=False)
            self.model.rho   = tf.Variable(loss_params['init_rho'], trainable=False)
            self.use_dynamic_alpha_rho = True
        except:
            self.model.alpha = loss_params['alpha']
            self.model.rho   = loss_params['rho']
            self.use_dynamic_alpha_rho = False
        self.model.W = self.W
        my_loss = lambda x,y: NOTEARS_loss(x, y,
                                           loss_params['l1'],
                                           self.model.alpha,
                                           self.model.rho)
        self.model.compile(loss=my_loss,
                     optimizer=self.opt,
                     metrics = self.metrics)


    def fit(self, C, X, epochs, batch_size, es_patience=None, val_split=0.25, callbacks=[], verbose=1):
        if es_patience is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience))
        if self.use_dynamic_alpha_rho:
            callbacks.append(DynamicAlphaRho(C))
        if verbose:
            bar = 'NOTEARS {l_bar}{bar} {n_fmt}/{total_fmt} ETA: {remaining}s,  {rate_fmt}{postfix}' 
            callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False, overall_bar_format=bar))
        self.model.fit(C, X, batch_size=batch_size, epochs=epochs,
            callbacks=callbacks, validation_split=val_split, verbose=0)


    def predict_w(self, C, project_to_dag=False):
        if project_to_dag:
            my_w = graph_utils.project_to_dag(self.get_w())[0]
            return np.array([my_w for _ in range(len(C))])
        else:
            return np.array([self.get_w() for _ in range(len(C))])


    def set_w(self, W):
        self.W.W.assign(W)

    def get_w(self):
        return self.W.W.numpy()
