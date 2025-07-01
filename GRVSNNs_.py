# GRVSNNs.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, Multiply, Add, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from bayes_opt import BayesianOptimization

## Important: Only enable eager execution if you need it for debugging.
# tf.config.run_functions_eagerly(True)

# define the Layers (these classes remain global for easy access by functions)
class GatedResidualUnit(Layer):
    def __init__(self, units, **kwargs):
        super(GatedResidualUnit, self).__init__(**kwargs)
        self.units = units
        self.dense_f1 = Dense(units, activation='elu', kernel_regularizer=l2(0.01))
        self.dense_h = Dense(units, activation='linear', kernel_regularizer=l2(0.01))
        self.dense_g = Dense(units, activation='sigmoid', kernel_regularizer=l2(0.01))
        self.layer_norm = LayerNormalization()

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.residual_projection = Dense(self.units, activation='linear', kernel_regularizer=l2(0.01))
        else:
            self.residual_projection = tf.identity
        super(GatedResidualUnit, self).build(input_shape)

    def call(self, inputs):
        residual = inputs
        f1 = self.dense_f1(inputs)
        f2 = f1
        h = self.dense_h(f2)
        g = self.dense_g(f2)
        gated_h = Multiply()([h, g])
        projected_residual = self.residual_projection(residual)
        gated_residual = Multiply()([projected_residual, (1 - g)])
        z = Add()([gated_h, gated_residual])
        hat_z = self.layer_norm(z)
        return hat_z

class VariableSelectionNetwork(Layer):
    def __init__(self, units, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.units = units
        self.dense_scores = Dense(units, activation='linear', kernel_regularizer=l2(0.01))

        # --- Hard-Sigmoid based Feature Selection (Commented out, for alternative use) ---
        # self.dense_hard_sigmoid_scores = Dense(units, activation='sigmoid', kernel_regularizer=l2(0.01))

    def call(self, inputs):
        inputs_for_selection = inputs

        # --- Softmax-based Feature Selection (active) ---
        raw_scores = self.dense_scores(inputs_for_selection)
        weights_w_i = tf.nn.softmax(raw_scores, axis=-1)
        selected_features = Multiply()([inputs_for_selection, weights_w_i])

        # Hard-sigmoid part
        # if you want to use the hard-sigmoid for feature selection, the steps are as:
        # 1. uncomment the 'self.dense_hard_sigmoid_scores' in __init__.
        # 2. uncomment the following block and comment out the softmax block above.
        # 3. adjust the feature importance calculation in the training loop accordingly.
        #
        # # if inputs.shape[1] < self.units:
        # #     inputs_for_selection_hard_sigmoid = inputs
        # # else:
        # #     inputs_for_selection_hard_sigmoid = inputs[:, :self.units]
        # # dense_output_hard_sigmoid = self.dense_hard_sigmoid_scores(inputs_for_selection_hard_sigmoid)
        # # hard_sigmoid_output = tf.maximum(0.0, tf.minimum(1.0, 0.2 * dense_output_hard_sigmoid + 0.5))
        # # selected_features = Multiply()([inputs_for_selection_hard_sigmoid, hard_sigmoid_output])

        return selected_features

# Create the GRVSNN model function
def create_grn_vsn_model(input_shape, dropout_rate=0.3, n_hidden_units=128):
    inputs = Input(shape=input_shape)
    grn_output = GatedResidualUnit(units=n_hidden_units)(inputs)
    grn_output = Dropout(dropout_rate)(grn_output)
    selected_features = VariableSelectionNetwork(units=n_hidden_units)(grn_output)
    outputs = Dense(2, activation='linear')(selected_features)
    model = Model(inputs, outputs)
    return model

# Main function to encapsulate the training and evaluation logic
def run_grvsn_training(loadings_file_path, micedata_file_path):
    # read data
    loadings_df = pd.read_csv(loadings_file_path)
    micedata_df = pd.read_csv(micedata_file_path)

    # data processing
    micedata_features = micedata_df.iloc[:, 2:].values
    loadings_real = loadings_df.map(lambda x: np.real(complex(x)) if isinstance(x, str) else np.real(x)).values

    print(f"Shape of micedata_features: {micedata_features.shape}")
    print(f"Shape of loadings_real: {loadings_real.shape}")

    # feature reduction
    loadings_reduced = loadings_real

    # combine features
    combined_features = np.hstack((micedata_features, loadings_reduced))
    print(f"Shape of combined_features after concatenation: {combined_features.shape}")

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features)
    target = micedata_df.iloc[:, :2].values

    # BO for hyperparameters
    def train_model(lr, dropout_rate):
        lr = max(0.00001, min(lr, 0.001))
        dropout_rate = max(0.1, min(dropout_rate, 0.5))

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = []

        for train_index, val_index in kf.split(normalized_features):
            X_train, X_val = normalized_features[train_index], normalized_features[val_index]
            y_train, y_val = target[train_index], target[val_index]

            model = create_grn_vsn_model(input_shape=(normalized_features.shape[1],), dropout_rate=dropout_rate, n_hidden_units=128)
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

            loss, _ = model.evaluate(X_val, y_val, verbose=0)
            mse_scores.append(loss)
        return -np.mean(mse_scores)

    # BO setup
    optimizer = BayesianOptimization(f=train_model, pbounds={'lr': (0.00001, 0.001), 'dropout_rate': (0.1, 0.5)}, verbose=2)
    optimizer.maximize(init_points=5, n_iter=10)

    # Training and test with best parameters
    best_params = optimizer.max['params']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    final_mse_scores = []
    trained_model = None # To store the last trained model for feature importance

    for fold, (train_index, val_index) in enumerate(kf.split(normalized_features)):
        X_train, X_val = normalized_features[train_index], normalized_features[val_index]
        y_train, y_val = target[train_index], target[val_index]

        final_model = create_grn_vsn_model(input_shape=(normalized_features.shape[1],), dropout_rate=best_params['dropout_rate'], n_hidden_units=128)
        final_model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='mean_squared_error',
                            metrics=[tf.keras.metrics.MeanSquaredError(name='mse_col1'), tf.keras.metrics.MeanSquaredError(name='mse_col2')])

        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        print(f"\n--- Fold {fold + 1} ---")
        final_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping])

        loss, mse_col1, mse_col2 = final_model.evaluate(X_val, y_val, verbose=0)
        final_mse_scores.append(loss)
        print(f"Fold test MSE: {loss}\nTrait 1 test MSE: {mse_col1}\nTrait 2 test MSE: {mse_col2}")

    print(f"\nAverage Test MSE across 5 folds: {np.mean(final_mse_scores)}")

    # Optional: feature importance calculation (done on the last trained model)
    try:
        if trained_model is not None: # Ensure a model was trained
            vsn_layer = None
            for layer in trained_model.layers:
                if isinstance(layer, VariableSelectionNetwork):
                    vsn_layer = layer
                    break

            if vsn_layer is not None:
                gru_output_layer = None
                for i, layer in enumerate(trained_model.layers):
                    if isinstance(layer, GatedResidualUnit):
                        gru_output_layer = trained_model.layers[i].output
                        break

                if gru_output_layer is not None:
                    grn_model_for_hat_z = Model(inputs=trained_model.input, outputs=gru_output_layer)
                    sample_hat_z = grn_model_for_hat_z.predict(normalized_features[:10])

                    vsn_raw_scores = vsn_layer.dense_scores(sample_hat_z).numpy()
                    vsn_weights_w_i = tf.nn.softmax(vsn_raw_scores, axis=-1).numpy()

                    print("\n--- Example Feature Importance (Softmax Weights from VSN) ---")
                    print(f"Average weights (across first 10 samples) for each feature in the VS block output:\n{np.mean(vsn_weights_w_i, axis=0)}")

                    vsn_output_example = vsn_layer(sample_hat_z).numpy()
                    threshold = 0.05
                    num_selected_features_per_sample = np.sum(vsn_output_example >= threshold, axis=1)
                    print(f"\nNumber of features with z'_i >= {threshold} (per sample, first 10 samples):")
                    print(num_selected_features_per_sample)
                else:
                    print("Could not find GatedResidualUnit layer to extract hat_z.")
            else:
                print("Could not find VariableSelectionNetwork layer.")
        else:
            print("No model was trained to perform feature importance analysis.")

    except Exception as e:
        print(f"\nCould not perform feature importance analysis due to an error: {e}")
        print("Ensure the model is trained and layer names/indices are correct.")

    # Return results if needed, e.g., average MSE
    return {
        "average_test_mse": np.mean(final_mse_scores),
        "best_hyperparameters": best_params
    }

# This block ensures run_grvsn_training() only runs when the script is executed directly
# from Python, not when it's imported as a module by R.
if __name__ == "__main__":
    # Example usage if you run this script directly in Python
    # please replace with your own data paths
    loadings_path = ''
    micedata_path = ''
    results = run_grvsn_training(loadings_path, micedata_path)
    print("\n--- Training Completed ---")
    print(f"Overall Average Test MSE: {results['average_test_mse']}")
    print(f"Best Hyperparameters: {results['best_hyperparameters']}")
