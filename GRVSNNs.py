{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wSpRq2SBzKUo",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 20367,
     "status": "ok",
     "timestamp": 1743592223512,
     "user": {
      "displayName": "hua Yu",
      "userId": "07626386458469956541"
     },
     "user_tz": -180
    },
    "id": "wSpRq2SBzKUo",
    "outputId": "6f0d9ebc-31a2-42b6-d5b9-76e0bb7e8a3c"
   },
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "WvOzqR-hM1Zg",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 4633,
     "status": "ok",
     "timestamp": 1743592343632,
     "user": {
      "displayName": "hua Yu",
      "userId": "07626386458469956541"
     },
     "user_tz": -180
    },
    "id": "WvOzqR-hM1Zg",
    "outputId": "9b0c984e-59b4-42b2-e1d1-1ae36597653f"
   },
   "outputs": [],
   "source": [
    ""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "gsO3a2mUqLIT",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 480
    },
    "executionInfo": {
     "elapsed": 69591,
     "status": "error",
     "timestamp": 1743592441296,
     "user": {
      "displayName": "hua Yu",
      "userId": "07626386458469956541"
     },
     "user_tz": -180
    },
    "id": "gsO3a2mUqLIT",
    "outputId": "abef6860-b0ed-4da4-db1f-8bcb50f3b0c1"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.model_selection import train_test_split, KFold\n",
    "from tensorflow.keras.layers import Input, Dense, Layer, Multiply, Add, Dropout\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "from tensorflow.keras.regularizers import l2\n",
    "\n",
    "\n",
    "import tensorflow as tf\n",
    "tf.config.run_functions_eagerly(True)\n",
    "\n",
    "import tensorflow as tf\n",
    "from bayes_opt import BayesianOptimization\n",
    "\n",
    "# 1. Read Data\n",
    "loadings_df = pd.read_csv('  ') #the path for the loadings file\n",
    "micedata_df = pd.read_csv('  ') ##the path for the genomic data\n",
    "\n",
    "# 2. Data Processing\n",
    "micedata_features = micedata_df.iloc[:, 2:].values\n",
    "loadings_real = loadings_df.applymap(lambda x: np.real(complex(x)) if isinstance(x, str) else np.real(x)).values\n",
    "\n",
    "# 3. Feature Reduction using PCA\n",
    "pca_micedata = PCA(n_components=169)\n",
    "micedata_features_reduced = pca_micedata.fit_transform(micedata_features)\n",
    "pca_loadings = PCA(n_components=169)\n",
    "loadings_real_reduced = pca_loadings.fit_transform(loadings_real)\n",
    "\n",
    "# 4. Combine Features\n",
    "combined_features = np.dot(micedata_features_reduced, loadings_real_reduced.T)\n",
    "scaler = StandardScaler()\n",
    "normalized_features = scaler.fit_transform(combined_features)\n",
    "target = micedata_df.iloc[:, :2].values\n",
    "\n",
    "# 5. Define Custom Layers\n",
    "class GatedResidualUnit(Layer):\n",
    "    def __init__(self, units, **kwargs):\n",
    "        super(GatedResidualUnit, self).__init__(**kwargs)\n",
    "        self.units = units\n",
    "        self.dense = Dense(units, activation='elu', kernel_regularizer=l2(0.01))\n",
    "        self.gate = Dense(units, activation='sigmoid', kernel_regularizer=l2(0.01))\n",
    "\n",
    "    def call(self, inputs):\n",
    "        residual = inputs\n",
    "        x = self.dense(inputs)\n",
    "        gate = self.gate(inputs)\n",
    "        return Add()([Multiply()([x, gate]), residual])\n",
    "\n",
    "class VariableSelectionNetwork(Layer):\n",
    "    def __init__(self, units, **kwargs):\n",
    "        super(VariableSelectionNetwork, self).__init__(**kwargs)\n",
    "        self.dense = Dense(units, activation='sigmoid', kernel_regularizer=l2(0.01))\n",
    "\n",
    "    def call(self, inputs):\n",
    "        inputs_reduced = inputs[:, :128]\n",
    "        dense_output = self.dense(inputs_reduced)\n",
    "        hard_sigmoid_output = tf.maximum(0.0, tf.minimum(1.0, 0.2 * dense_output + 0.5))\n",
    "        return Multiply()([inputs_reduced, hard_sigmoid_output])\n",
    "\n",
    "# 6. Create Model Function\n",
    "def create_grn_vsn_model(input_shape, dropout_rate=0.3):\n",
    "    inputs = Input(shape=input_shape)\n",
    "    selected_features = VariableSelectionNetwork(units=128)(inputs)\n",
    "    selected_features = Dropout(dropout_rate)(selected_features)\n",
    "    grn_output = GatedResidualUnit(units=128)(selected_features)\n",
    "    outputs = Dense(2)(grn_output)\n",
    "    model = Model(inputs, outputs)\n",
    "    return model\n",
    "\n",
    "# 7. Bayesian Optimization for Hyperparameters\n",
    "def train_model(lr, dropout_rate):\n",
    "    lr = max(0.00001, min(lr, 0.001))\n",
    "    dropout_rate = max(0.1, min(dropout_rate, 0.5))\n",
    "\n",
    "    kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
    "    mse_scores = []\n",
    "\n",
    "    for train_index, val_index in kf.split(normalized_features):\n",
    "        X_train, X_val = normalized_features[train_index], normalized_features[val_index]\n",
    "        y_train, y_val = target[train_index], target[val_index]\n",
    "\n",
    "        model = create_grn_vsn_model(input_shape=(normalized_features.shape[1],), dropout_rate=dropout_rate)\n",
    "        # Create a new optimizer instance for each model\n",
    "        optimizer = Adam(learning_rate=lr)\n",
    "        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])\n",
    "\n",
    "        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)\n",
    "        model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)\n",
    "\n",
    "        loss, _ = model.evaluate(X_val, y_val, verbose=0)\n",
    "        mse_scores.append(loss)\n",
    "\n",
    "    return -np.mean(mse_scores)\n",
    "\n",
    "optimizer = BayesianOptimization(f=train_model, pbounds={'lr': (0.00001, 0.001), 'dropout_rate': (0.1, 0.5)}, verbose=2)\n",
    "optimizer.maximize(init_points=5, n_iter=10)\n",
    "\n",
    "# 8. Train Best Model with Cross-Validation\n",
    "best_params = optimizer.max['params']\n",
    "kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "final_mse_scores = []\n",
    "for train_index, val_index in kf.split(normalized_features):\n",
    "    X_train, X_val = normalized_features[train_index], normalized_features[val_index]\n",
    "    y_train, y_val = target[train_index], target[val_index]\n",
    "\n",
    "    final_model = create_grn_vsn_model(input_shape=(normalized_features.shape[1],), dropout_rate=best_params['dropout_rate'])\n",
    "    final_model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='mean_squared_error',\n",
    "                        metrics=[tf.keras.metrics.MeanSquaredError(name='mse_col1'), tf.keras.metrics.MeanSquaredError(name='mse_col2')])\n",
    "\n",
    "    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)\n",
    "    final_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping])\n",
    "\n",
    "    loss, mse_col1, mse_col2 = final_model.evaluate(X_val, y_val)\n",
    "    final_mse_scores.append(loss)\n",
    "    print(f\"Fold test MSE: {loss}\\nTrait 1 test MSE: {mse_col1}\\nTrait 2 test MSE: {mse_col2}\")\n",
    "\n",
    "print(f\"Average Test MSE across 5 folds: {np.mean(final_mse_scores)}\")"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python (myenv)",
   "language": "python",
   "name": "myenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
