import os
import queue
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.sparse as sp
import tensorflow_probability as tfp
from tqdm import tqdm
from utils import csr_to_sparse_tensor
from scipy.stats import spearmanr


class LTRModel:
    def __init__(self, order, rank,
                 multirank=None,
                 nepochs=100,
                 batch_size=500,
                 regularization_factor=1e-7,
                 loss_fn=tf.keras.losses.MSE,
                 optimizer=None,
                 show_progress=True,
                 early_stopping_patience=0,
                 rank_step=None,
                 logging=False):

        """
        Initialize a new LTR model object
        :param order: the order of the polynomial
        :param rank:
        :param multirank:
        :param nepochs:
        :param batch_size:
        :param regularization_factor:
        :param loss_fn:
        :param optimizer:
        :param show_progress:
        :param early_stopping_patience:
        :param rank_step:
        :param logging:
        """

        self.order = order
        self.rank = rank
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.regularization_factor = regularization_factor
        self.loss_fn = loss_fn
        self.show_progress = show_progress
        self.logging = logging
        self.nfeatures = None
        self.sparse = False

        #  If multirank is not given, train all ranks simultaneously
        if multirank is None:
            self.multirank = rank
        else:
            self.multirank = multirank

        #  If rank step is undefined, progress by increments of multirank
        if rank_step is None:
            self.rank_step = self.multirank
        else:
            self.rank_step = rank_step

        #  Start training at rank zero
        self.current_training_rank = 0

        #  Default optimizer is Adam with learning rate 0.01
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = keras.optimizers.Adam(0.01)

        #  Set up early stopping if used
        if early_stopping_patience:
            self.early_stopping = True
            self.patience = early_stopping_patience
        else:
            self.early_stopping = False
            self.patience = None

        #  Placeholders for the weight vector and P matrices
        self.lambda_vec = None
        self.P_list = None

        #  Lambda function which returns the model parameters that is used with autograd
        self.var_list_fn = lambda: [self.lambda_vec] + self.P_list

        #  Keep track of the deflated output
        self.y_deflated = None

    def init_vars(self):
        """Initialize model parameters"""
        self.lambda_vec = tf.Variable(tf.ones(shape=(self.rank, 1), dtype=tf.float64) / self.rank,
                                      name='lambda')
        self.P_list = [tf.Variable(tf.random.uniform(shape=(self.rank, self.nfeatures), dtype=tf.float64),
                                   name=f'P_{i}') for i in range(self.order)]

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, _X, lambda_vec, *P_list):
        """
        Evaluate the model on data _X
        :param _X: observations to be fed into the model
        :param lambda_vec: rank weight parameter vector
        :param P_list: list of P matrices
        :return: column vector of model outputs on the inputs
        """
        if self.sparse:
            mm = tf.sparse.sparse_dense_matmul
        else:
            mm = tf.matmul

        #  Compute the XP^T matrices
        constituents = [mm(_X, tf.transpose(P)) for P in P_list]
        #  Compute the F matrix
        F = tf.math.reduce_prod(tf.stack(constituents, axis=2), axis=2)
        #  Multiply by lambda but don't sum reduce
        pre_F = F * tf.transpose(lambda_vec)

        return tf.expand_dims(tf.reduce_sum(pre_F, axis=1), axis=-1)

    @tf.function
    def compute_reg(self):
        """Compute the regularization term"""
        return self.regularization_factor / (self.rank * self.order * self.nfeatures) * sum(
            [tf.reduce_sum(tf.norm(P, axis=1)) for P in self.P_list])

    def numpy_to_tensor(self, X):
        """Construct a Tensorflow tensor from a numpy array"""
        if self.sparse:
            X_t = csr_to_sparse_tensor(X)
        else:
            X_t = tf.convert_to_tensor(X)
        return X_t

    def numpy_to_dataset(self, X):
        """Construct a Tensorflow dataset from a numpy array"""
        X_t = self.numpy_to_tensor(X)
        return tf.data.Dataset.from_tensor_slices(X_t)

    # @tf.function
    def predict(self, X_dataset, include_y=False):
        """
        Use model to predict output on input X_dataset
        :param X_dataset: observations whose outputs are predicted
        :param include_y: whether the input dataset contains y
        :return: model prediction on the inputs
        """
        predictions = tf.TensorArray(tf.float64,
                                     size=1,
                                     dynamic_size=True,
                                     infer_shape=False,
                                     element_shape=tf.TensorShape([None, 1]),
                                     clear_after_read=True)

        #  Compute prediction in mini-batches
        if include_y:
            for i, (_X, _y) in enumerate(X_dataset.batch(self.batch_size)):
                # print(type(_X))
                prediction = self(_X, self.lambda_vec, *self.P_list)
                predictions.write(i, prediction).mark_used()
        else:
            for i, _X in enumerate(X_dataset.batch(self.batch_size)):
                # print(type(_X))
                prediction = self(_X, self.lambda_vec, *self.P_list)
                predictions.write(i, prediction).mark_used()
        #  Collect mini-batch outputs
        return predictions.concat()

    # @tf.function
    def partial_predict(self, X_dataset, lambda_vec, P_list):
        """
        Use only some ranks to predict output on the data X_dataset
        :param X_dataset: observations whose outputs are predicted
        :param lambda_vec: rank weight vector
        :param P_list: list of P matrices
        :return: partial model prediction on the inputs
        """
        predictions = tf.TensorArray(tf.float64,
                                     size=1,
                                     dynamic_size=True,
                                     infer_shape=False,
                                     element_shape=tf.TensorShape([None, 1]),
                                     clear_after_read=True)

        #  Compute prediction in mini-batches
        for i, (_X, _y) in enumerate(X_dataset.batch(batch_size=self.batch_size)):
            prediction = self(_X, lambda_vec, *P_list)
            predictions.write(i, prediction).mark_used()
        #  Collect mini-batch outputs
        return predictions.concat()

    def predict_numpy(self, X):
        """Predict outputs for observations in a numpy array"""
        X_dataset = self.numpy_to_dataset(X)
        return self.predict(X_dataset)

    def initialize_writer(self, run_name):
        """Initialize logging for Tensorboard"""
        log_dir = "./logs/"

        #  Find unused name for log file
        i = 1  # Index runs that use the same configuration starting from one
        while os.path.exists(log_dir + run_name + f"_{i}"):
            i += 1
        writer = tf.summary.create_file_writer(log_dir + run_name + f"_{i}")
        writer.init()

        return writer

    @tf.function
    def training_step(self, _X, _y):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            #  Watch the model parameters
            tape.watch(self.var_list_fn())

            #  If training all ranks simultaneously, use full P matrices
            if self.rank == self.multirank:
                P_list = self.P_list
                lambda_vec = self.lambda_vec
            else:
                P_list = [P[self.current_training_rank: self.current_training_rank + self.multirank, :] for P in
                          self.P_list]
                lambda_vec = self.lambda_vec[self.current_training_rank: self.current_training_rank + self.multirank]

            #  Forward pass
            output = self(_X, lambda_vec, *P_list)

            #  Compute loss and regularization
            loss = self.loss_fn(tf.transpose(_y), tf.transpose(output)) + self.compute_reg()

            #  Compute gradient
            grads = tape.gradient(loss, self.var_list_fn())

            #  Apply gradients
            self.optimizer.apply_gradients(zip(grads, self.var_list_fn()))

    def deflate_output(self, X_dataset):
        a = self.current_training_rank
        b = self.current_training_rank + self.rank_step

        lambda_deflate = self.lambda_vec[a:b]
        P_deflate = [P[a:b, :] for P in self.P_list]

        self.y_deflated = self.y_deflated - self.partial_predict(X_dataset, lambda_deflate, P_deflate)

    def fit(self, X_train, y_train, X_test, y_test, run_name):

        if self.nfeatures is None:
            self.nfeatures = X_train.shape[1]

        #  Whether the inputs are given as sparse arrays
        self.sparse = sp.issparse(X_train)
        self.init_vars()

        #  If input is numpy array or sparse array, then convert to Tensorflow tensor
        if isinstance(X_train, (np.ndarray, sp.csr.csr_matrix)):
            X_train = self.numpy_to_tensor(X_train)
            y_train = tf.convert_to_tensor(y_train)

            X_test = self.numpy_to_tensor(X_test)
            y_test = tf.convert_to_tensor(y_test)

        #  Initialize early stopping queue
        if self.early_stopping:
            early_stopping_queue = queue.Queue(self.patience)
            early_stopping_queue.put(np.inf)
        else:
            early_stopping_queue = None

        #  Initialize logging for Tensorboard
        if self.logging:
            writer = self.initialize_writer(run_name)
        else:
            writer = tf.summary.create_noop_writer()

        self.y_deflated = y_train.numpy().copy()

        #  If we are computing test statistics or use early stopping, then we need the test dataset
        use_test_set = self.logging or self.early_stopping

        if use_test_set:
            test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
        else:
            test_dataset = None

        step_index = 0

        with writer.as_default():
            while self.current_training_rank + self.multirank <= self.rank:
                #  Rebuild the training dataset using the deflated y
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, self.y_deflated)).shuffle(
                    buffer_size=500000,
                    reshuffle_each_iteration=True)

                for epoch in tqdm(range(self.nepochs), unit='epoch', disable=(not self.show_progress)):
                    for i, (_X, _y) in enumerate(train_dataset.batch(batch_size=self.batch_size)):
                        #  Make one training step using the minibatch
                        self.training_step(_X, _y)

                    if use_test_set:
                        #  Predict y on the test set
                        test_prediction = self.predict(test_dataset)

                        #  Compute summary statistics
                        test_pearson = tf.reduce_mean(tfp.stats.correlation(y_test, test_prediction))
                        test_rmse = tf.sqrt(tf.reduce_mean(tf.square(y_test - test_prediction)))
                        test_spearman = spearmanr(y_test, test_prediction)[0]

                        #  Log summary statistics
                        tf.summary.scalar("Test RMSE", test_rmse, step=step_index)
                        tf.summary.scalar("Test Pearson", test_pearson, step=step_index)
                        tf.summary.scalar("Test Spearman", test_spearman, step=step_index)

                        #  Also log learning rate to verify scheduler function
                        tf.summary.scalar("learning rate", self.optimizer._decayed_lr(tf.float64), step=step_index)

                        #  Increment the logging step counter and flush the Tensorboard writer
                        step_index += 1
                        writer.flush()

                        if self.early_stopping:
                            #  Compute percentage change of test RMSE from previous epochs
                            change_to_previous = (np.array(early_stopping_queue.queue) - test_rmse) / test_rmse
                            #  If test rmse does not improve by 0.01% for the specified number of epochs, then break
                            if np.all(change_to_previous < 0.01) and epoch > self.patience:
                                self.deflate_output(X_train)
                                self.current_training_rank = self.current_training_rank + self.rank_step
                                break

                            #  If early stopping queue is full, then remove the oldest element
                            if early_stopping_queue.full():
                                _ = early_stopping_queue.get()
                            early_stopping_queue.put(test_rmse)

                #  Deflate the training y using the outgoing ranks
                self.deflate_output(train_dataset)
                #  Increment training rank by rank_step
                self.current_training_rank = self.current_training_rank + self.rank_step
