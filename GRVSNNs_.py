# GRVSNNs.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.layers import Input, Dense, Layer, Multiply, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from bayes_opt import BayesianOptimization
import tensorflow as tf

tf.config.run_functions_eagerly(True)

# Custom layers
class GatedResidualUnit(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = Dense(units, activation='elu', kernel_regularizer=l2(0.01))
        self.gate = Dense(units, activation='sigmoid', kernel_regularizer=l2(0.01))
        self.residual_projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.residual_projection = Dense(self.units, activation='linear', kernel_regularizer=l2(0.01))
        else:
            self.residual_projection = tf.identity
        super().build(input_shape)

    def call(self, inputs):
        residual = inputs
        x = self.dense(inputs)
        gate = self.gate(inputs)
        gated_output = Multiply()([x, gate])
        projected_residual = self.residual_projection(residual) if self.residual_projection is not tf.identity else residual
        return Add()([gated_output, projected_residual])

class VariableSelectionNetwork(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = Dense(units, activation='sigmoid', kernel_regularizer=l2(0.01))

    def call(self, inputs):
        if inputs.shape[1] < self.units:
            inputs_for_selection = inputs
        else:
            inputs_for_selection = inputs[:, :self.units]

        dense_output = self.dense(inputs_for_selection)
        hard_sigmoid_output = tf.maximum(0.0, tf.minimum(1.0, 0.2 * dense_output + 0.5))
        return Multiply()([inputs_for_selection, hard_sigmoid_output])

def create_grn_vsn_model(input_shape, dropout_rate=0.3, n_hidden_units=128):
    inputs = Input(shape=input_shape)
    grn_output = GatedResidualUnit(units=n_hidden_units)(inputs)
    grn_output = Dropout(dropout_rate)(grn_output)
    selected_features = VariableSelectionNetwork(units=n_hidden_units)(grn_output)
    outputs = Dense(2)(selected_features)
    return Model(inputs, outputs)

def run_training_pipeline(loadings_path, micedata_path):
    # Read data
    loadings_df = pd.read_csv(loadings_path)
    micedata_df = pd.read_csv(micedata_path)

    micedata_features = micedata_df.iloc[:, 2:].values
    loadings_real = loadings_df.applymap(lambda x: np.real(complex(x)) if isinstance(x, str) else np.real(x)).values
    combined_features = np.hstack((micedata_features, loadings_real))

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features)
    target = micedata_df.iloc[:, :2].values

    def train_model(lr, dropout_rate):
        lr = max(0.00001, min(lr, 0.001))
        dropout_rate = max(0.1, min(dropout_rate, 0.5))

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = []

        for train_index, val_index in kf.split(normalized_features):
            X_train, X_val = normalized_features[train_index], normalized_features[val_index]
            y_train, y_val = target[train_index], target[val_index]

            model = create_grn_vsn_model(input_shape=(normalized_features.shape[1],), dropout_rate=dropout_rate)
            model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val),
                      callbacks=[early_stopping], verbose=0)
            loss, _ = model.evaluate(X_val, y_val, verbose=0)
            mse_scores.append(loss)

        return -np.mean(mse_scores)

    optimizer = BayesianOptimization(f=train_model, pbounds={'lr': (0.00001, 0.001), 'dropout_rate': (0.1, 0.5)}, verbose=2)
    optimizer.maximize(init_points=5, n_iter=10)

    best_params = optimizer.max['params']
    final_mse_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(normalized_features):
        X_train, X_val = normalized_features[train_index], normalized_features[val_index]
        y_train, y_val = target[train_index], target[val_index]

        final_model = create_grn_vsn_model(input_shape=(normalized_features.shape[1],),
                                           dropout_rate=best_params['dropout_rate'])
        final_model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='mean_squared_error',
                            metrics=[tf.keras.metrics.MeanSquaredError(name='mse_col1'),
                                     tf.keras.metrics.MeanSquaredError(name='mse_col2')])
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        final_model.fit(X_train, y_train, epochs=100, batch_size=128,
                        validation_data=(X_val, y_val), callbacks=[early_stopping])
        loss, mse_col1, mse_col2 = final_model.evaluate(X_val, y_val)
        final_mse_scores.append(loss)

    return np.mean(final_mse_scores)
