from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from autoPyTorch import AutoNetRegression
import autokeras as ak
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, balanced_accuracy_score, accuracy_score
import keras
from keras.layers import Dense, LSTM, Dropout, Reshape
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import Input, Model
from keras.optimizers import Adam
import numpy as np


class AutoLioNet:

    def __init__(self, encoded_x_train, x_train, time=None):

        self.encoded_x_train = encoded_x_train
        self.x_train = x_train
        self.time = time

    def best_strategy(self, encoded_x_train, x_train, time):

        #Best decoders of all strategies
        classic_ml = self._classic_ml_strategy(encoded_x_train, x_train)
        simple_nn = self._simple_nn_strategy(encoded_x_train, x_train)
        autokeras = self._autokeras_strategy(encoded_x_train, x_train, time)
        autopytorch = self._autopytorch_strategy(encoded_x_train, x_train, time)
        #Compute mae
        scores = []
        classic_ml_score = mean_absolute_error(x_train, classic_ml.predict(encoded_x_train))
        scores.append(classic_ml_score)
        simple_nn_scores = simple_nn.evaluate(encoded_x_train, x_train)
        simple_nn_score = simple_nn_scores[0]
        scores.append(simple_nn_score)
        autokeras_scores = autokeras.evaluate(encoded_x_train, x_train)[0]
        autokeras_score = autokeras_scores[0]
        scores.append(autokeras_score)
        autopytorch_score = mean_absolute_error(x_train, autopytorch.predict(encoded_x_train))
        scores.append(autopytorch_score)
        best_score = np.argmin(scores)
        #Compare mae
        if best_score == 0:
            return classic_ml
        elif best_score == 1:
            return simple_nn
        elif best_score == 2:
            return autokeras
        else:
            return autopytorch
 
    def _classic_ml_strategy(self, encoded_x_train, x_train):

        best_alpha = None
        best_score = 100
        alphas = [0.0000001, 0.00001, 0.001, 0.1, 0.5, 0.9, 1, 10, 100]
        for a in alphas:
            temp_model = MultiOutputRegressor(Ridge(alpha=a, random_state=123)).fit(encoded_x_train, x_train)
            y_pred = temp_model.predict(encoded_x_train)
            score = mean_absolute_error(x_train, y_pred)
            if score < best_score:
                best_score = score
                classicml_decoder = temp_model
        return classicml_decoder


    def _simple_nn_strategy(self, encoded_x_train, x_train):

        check_point = ModelCheckpoint("auto_dummy.hdf5", monitor="val_loss", verbose=2,save_best_only=True, mode="auto")
        input_dim = len(encoded_x_train[0])
        output_dim = len(x_train[0])
        main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')
        x = Reshape((1,input_dim))(main_input)
        x = LSTM(input_dim,activation='tanh')(x)
        x = Dropout(0.75)(x)
        output_lay = Dense(output_dim, activation='sigmoid')(x)
        auto_dummy = Model(inputs=[main_input], outputs=[output_lay])
        auto_dummy.compile(optimizer="adam",loss=['mean_absolute_error'])
        auto_dummy.fit(encoded_x_train, x_train, validation_split=0.3, epochs=100)
        return auto_dummy

    def _autokeras_strategy(self, encoded_x_train, x_train, time):

        if time is None:
            #Dense model for no time limitation
            input_node = ak.Input()
            output_node = ak.DenseBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=100)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                            validation_split=0.3, epochs=1000)
            autodecoder_dense = auto_model.export_model()
            #RNN model for no time limitation
            input_node = ak.Input()
            output_node = ak.RNNBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=20)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                            validation_split=0.3, epochs=50)
            autodecoder_rnn = auto_model.export_model()
            #Best model for no time limitation
            score_dense = autodecoder_dense.evaluate(encoded_x_train, x_train)
            score_rnn = autodecoder_rnn.evaluate(encoded_x_train, x_train)
            if score_dense[0] < score_rnn[0]:
                return autodecoder_dense
            else:
                return autodecoder_rnn

        if time == "low":
            #Dense model for low time limitation
            input_node = ak.Input()
            output_node = ak.DenseBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=10)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                            validation_split=0.3, epochs=200)
            autodecoder_dense = auto_model.export_model()
            #RNN model for low time limitation
            input_node = ak.Input()
            output_node = ak.RNNBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=1)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                validation_split=0.3, epochs=10)
            autodecoder_rnn = auto_model.export_model()
            #Best model for low time limitation
            score_dense = autodecoder_dense.evaluate(encoded_x_train, x_train)
            score_rnn = autodecoder_rnn.evaluate(encoded_x_train, x_train)
            if score_dense[0] < score_rnn[0]:
                return autodecoder_dense
            else:
                return autodecoder_rnn
        if time == "medium":
            #Dense model for medium time limitation
            input_node = ak.Input()
            output_node = ak.DenseBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=20)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                            validation_split=0.3, epochs=300)
            autodecoder_dense = auto_model.export_model()
            #RNN model for medium time limitation
            input_node = ak.Input()
            output_node = ak.RNNBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=5)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                            validation_split=0.3, epochs=50)
            autodecoder_rnn = auto_model.export_model()
            #Best model for medium time limitation
            score_dense = autodecoder_dense.evaluate(encoded_x_train, x_train)
            score_rnn = autodecoder_rnn.evaluate(encoded_x_train, x_train)
            if score_dense[0] < score_rnn[0]:
                return autodecoder_dense
            else:
                return autodecoder_rnn
        if time == "high":
            #Dense model for high time limitation
            input_node = ak.Input()
            output_node = ak.DenseBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=50)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                            validation_split=0.3, epochs=500)
            autodecoder_dense = auto_model.export_model()
            #RNN model for high time limitation
            input_node = ak.Input()
            output_node = ak.RNNBlock()(input_node)
            output_node = ak.RegressionHead(activation='sigmoid', loss='mean_absolute_error')(output_node)
            auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=10)
            auto_model.fit(encoded_x_train.reshape((len(encoded_x_train),500,1)), x_train,
                            validation_split=0.3, epochs=50)
            autodecoder_rnn = auto_model.export_model()
            #Best model for high time limitation
            score_dense = autodecoder_dense.evaluate(encoded_x_train, x_train)
            score_rnn = autodecoder_rnn.evaluate(encoded_x_train, x_train)
            if score_dense[0] < score_rnn[0]:
                return autodecoder_dense
            else:
                return autodecoder_rnn

    def _autopytorch_strategy(self, encoded_x_train, x_train, time):
        #No time limitation
        if time is None:
            autonet = AutoNetRegression(config_preset="full_cs", result_logger_dir="logs2/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=210, 
                             max_budget=240, max_runtime=5400)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet
        #Low time category
        if time == "low":
            autonet = AutoNetRegression(config_preset="tiny_cs", result_logger_dir="logs/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=60,
                             max_budget=90, max_runtime=600)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet
        #Medium time category
        if time == "medium":
            autonet = AutoNetRegression(config_preset="medium_cs", result_logger_dir="logs1/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=120, 
                             max_budget=150, max_runtime=1800)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet
        #High time category
        if time == "high":
            autonet = AutoNetRegression(config_preset="full_cs", result_logger_dir="logs2/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=150, 
                             max_budget=180, max_runtime=3600)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet