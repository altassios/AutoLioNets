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
import pickle
import numpy as np
import tensorflow as tf


class AutoLioNet:

    def __init__(self, encoded_x_train, x_train, time=None, name="decoder"):

        self.encoded_x_train = encoded_x_train
        self.x_train = x_train
        self.time = time
        self.name = name
        self.best_strategy()

    def load_decoder(self, method=None):
        if method is not None:
            self.method == method
        if self.method == 0:
            decoder = pickle.load(open(self.name+'sav', 'rb'))
            print("Best model: classic ml model, saved:",self.name+'sav',)
            return decoder
        if self.method == 1 or self.method == 2:
            print("Best model: keras model, saved:",self.name)
            decoder = tf.keras.models.load_model(self.name)
            return decoder
        elif self.method == 3:
            print("Best model: torch model")
            return self.autopytorch_decoder


    def best_strategy(self):

        #Best decoders of all strategies
        classic_ml = self._classic_ml_strategy(self.encoded_x_train, self.x_train)
        simple_nn = self._simple_nn_strategy(self.encoded_x_train, self.x_train)
        autokeras = self._autokeras_strategy(self.encoded_x_train, self.x_train, self.time)
        autopytorch = self._autopytorch_strategy(self.encoded_x_train, self.x_train, self.time)
        #Compute mae
        scores = []
        classic_ml_score = mean_absolute_error(self.x_train, classic_ml.predict(self.encoded_x_train))
        scores.append(classic_ml_score)
        simple_nn_score = simple_nn.evaluate(self.encoded_x_train, self.x_train)
        scores.append(simple_nn_score)
        autokeras_scores = autokeras.evaluate(self.encoded_x_train, self.x_train)
        autokeras_score = autokeras_scores[0]
        scores.append(autokeras_score)
        autopytorch_score = mean_absolute_error(self.x_train, autopytorch.predict(self.encoded_x_train))
        scores.append(autopytorch_score)
        best_score = np.argmin(scores)
        #Compare mae
        filename = self.name+'1.sav'
        pickle.dump(classic_ml, open(filename, 'wb'))

        simple_nn.save(self.name+"2")

        autokeras.save(self.name+"3")

        self.autopytorch_decoder = autopytorch

        if best_score == 0:
            self.method = 0
        elif best_score == 1:
            self.method = 1
        elif best_score == 2:
            self.method = 2            
        else:
            self.method = 3
 
    def _classic_ml_strategy(self, encoded_x_train, x_train):

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
        elif time == "low":
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
        elif time == "medium":
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
        elif time == "high":
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
    
        if time is None: #No time limitation
            autonet = AutoNetRegression(config_preset="full_cs", result_logger_dir="logs2/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=210, 
                             max_budget=240, max_runtime=5400)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet
        elif time == "low": #Low time category
            autonet = AutoNetRegression(config_preset="tiny_cs", result_logger_dir="logs/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=60,
                             max_budget=90, max_runtime=600)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet
        elif time == "medium": #Medium time category
            autonet = AutoNetRegression(config_preset="medium_cs", result_logger_dir="logs1/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=120, 
                             max_budget=150, max_runtime=1800)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet
        elif time == "high": #High time category
            autonet = AutoNetRegression(config_preset="full_cs", result_logger_dir="logs2/", log_level='info',
                             optimize_metric = 'mean_abs_error', budget_type='time', min_budget=150, 
                             max_budget=180, max_runtime=3600)
            autonet.fit(X_train=encoded_x_train, Y_train=x_train, validation_split=0.3, refit=True)
            return autonet
