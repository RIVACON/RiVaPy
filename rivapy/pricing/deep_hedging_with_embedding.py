from typing import List, Dict, Union
import json
import numpy as np
import sys

try:
    import tensorflow as tf

    try:
        tf.config.run_functions_eagerly(False)
    except:
        pass
except:
    import warnings

    warnings.warn(
        "Tensorflow is not installed. You cannot use the PPA Deep Hedging Pricer!"
    )


class DeepHedgeModelwEmbedding(tf.keras.Model):
    def __init__(
        self,
        hedge_instruments: List[str],
        additional_states: List[str],
        timegrid: np.ndarray,
        regularization: float,
        depth: int,
        n_neurons: int,
        loss: str,
        transaction_cost: dict = None,
        # threshold: float = 0.0,
        no_of_unique_model: int = 1,
        # cascading: bool = False,
        model: tf.keras.Model = None,
        **kwargs
    ):
        """Class for Deep Hedging Model.

        Args:
            hedge_instruments (List[str]): List of keys of the instruments used for hedging. The keys in this list are used in the dictionary of paths.
            additional_states (List[str]): List of keys of additional states used for hedging. The keys in this list are used in the dictionary of paths.
            timegrid (np.ndarray): The timegrid of the paths. The model input gets the current time to maturity as an additional input that is computed at gridpoint t as (timegrid[-1]-timegrid[t])/self.timegrid[-1].
            regularization (float): The training of the hedge model is based on minimizing the loss function defined by the variance of the pnl minus this regularization term multiplied by the mean of the pnl (mean-variance optimal hedging)
            depth (int): If no model (neural network) is specified, the model is build as a fully connected neural network with this depth.
            n_neurons (int): If no model (neural network) is specified, the model is build as a fully connected neural network with this number of neurons per layer.
            loss (str): Determines whether mean variance or exponential utility are optimized.
            model (tf.keras.Model, optional): The model (neural network) used. If it is None, a fully connected neural network using the parameter depth and n_neurons. The network Input must equal the number of hedge instruments plus the number of additional states plus one input for the time to maturity. The output dimension must equal the number of hedge instruments. Defaults to None.
        """
        super().__init__(**kwargs)

        self.hedge_instruments = hedge_instruments
        if additional_states is None:
            self.additional_states = []
        else:
            self.additional_states = additional_states

            #self._concat_layer = tf.keras.layers.Concatenate(name="Concatenate")
        if model is None:
            if "emb_key" in self.additional_states:
                self.no_of_unique_model = no_of_unique_model
                self.embedding_size = 1#32
                self._embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.no_of_unique_model+1,
                    output_dim=self.embedding_size,
                    input_length=1,
                    name="Embedding",
                )
            self.model = self._build_model(depth, n_neurons)
        else:
            self.model = model
        self.timegrid = timegrid
        self.regularization = regularization
        self._prev_q = None
        self._forecast_ids = None
        self._loss = loss
        if transaction_cost is None:
            self.transaction_cost = {}
        else:
            self.transaction_cost = transaction_cost
        # self.threshold = threshold
        # self.cascading = cascading

    def __call__(self, x, training=True):
        if not self.transaction_cost:
            return self._compute_pnl(x, training)  # + self.price
        else:
            return self._compute_pnl_withconstains(x, training)  # + self.price
    

    def _build_model(self, depth: int, nb_neurons: int):
        inputs = [
            tf.keras.Input(shape=(1,), name=ins) for ins in self.hedge_instruments
        ]
        # if "emb_key" in self.additional_states:  #
        #if self.additional_states is not None:
         #   for state in self.additional_states:
         #       inp_cat_data = tf.keras.layers.Input(shape=(1,), name=state)
                #inputs.append(inp_cat_data)
        inputs.append(tf.keras.Input(shape=(1,), name="ttm"))

        if "emb_key" in self.additional_states:
            inp_cat_data = tf.keras.layers.Input(shape=(1,))
            inputs.append(inp_cat_data)
            emb = self._embedding_layer(inp_cat_data)
            flatten = tf.keras.layers.Flatten()(emb)
            fully_connected_Input1 = tf.keras.layers.concatenate(inputs)
            fully_connected_Input = tf.keras.layers.concatenate(
                    [fully_connected_Input1, flatten]
            )
        else:
            fully_connected_Input = tf.keras.layers.concatenate(inputs)

        values_all = tf.keras.layers.Dense(
            nb_neurons,
            activation="selu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(fully_connected_Input)
        for _ in range(depth):
            values_all = tf.keras.layers.Dense(
                nb_neurons,
                activation="selu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            )(values_all)
        value_out = tf.keras.layers.Dense(
            len(self.hedge_instruments),
            activation="linear",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(values_all)
        model = tf.keras.Model(inputs=inputs, outputs=value_out)
        return model


        inputs = [
            tf.keras.Input(shape=(1,), name=ins) for ins in self.hedge_instruments
        ]
        # if "emb_key" in self.additional_states:  #
        if self.additional_states is not None:
            for state in self.additional_states:
                inp_cat_data = tf.keras.layers.Input(shape=(1,), name=state)
                inputs.append(inp_cat_data)
        inputs.append(tf.keras.Input(shape=(1,), name="ttm"))

        if "emb_key" in self.additional_states:
            fully_connected_Input1 = tf.keras.layers.concatenate(inputs)
            emb = self._embedding_layer(inp_cat_data)
            flatten = tf.keras.layers.Flatten()(emb)
            fully_connected_Input = self._concat_layer(
                [fully_connected_Input1, flatten]
            )
        else:
            fully_connected_Input = tf.keras.layers.concatenate(inputs)

        values_all = tf.keras.layers.Dense(
            nb_neurons,
            activation="selu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(fully_connected_Input)
        for _ in range(depth):
            values_all = tf.keras.layers.Dense(
                nb_neurons,
                activation="selu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            )(values_all)
        value_out = tf.keras.layers.Dense(
            len(self.hedge_instruments),
            activation="linear",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(values_all)
        model = tf.keras.Model(inputs=inputs, outputs=value_out)
        return model
    

    @tf.function
    def _compute_pnl(self, x_in, training):
        if "emb_key" in self.additional_states:
            x = [x_in[0]]
            params = [x_in[1]]
        else:
            x = x_in
        pnl = tf.zeros((tf.shape(x[0])[0],))
        self._prev_q = tf.zeros(
            (tf.shape(x[0])[0], len(self.hedge_instruments)), name="prev_q"
        )
        for i in range(self.timegrid.shape[0] - 2):
            t = (
                [self.timegrid[-1] - self.timegrid[i]]
                * tf.ones((tf.shape(x[0])[0], 1))
                / self.timegrid[-1]
            )
            inputs = [v[:, i] for v in x]
            inputs.append(t)
            if "emb_key" in self.additional_states:
                #params = self._embedding_layer(x_in[1])
                inputs.append(params)
            quantity = self.model(inputs,training)#self.model(inputs, training=training)
            for j in range(len(self.hedge_instruments)):
                pnl += tf.math.multiply(
                    (self._prev_q[:, j] - quantity[:, j]), tf.squeeze(x[j][:, i])
                )
            self._prev_q = quantity
        for j in range(len(self.hedge_instruments)):
            pnl += self._prev_q[:, j] * tf.squeeze(x[j][:, -1])
        return pnl
    
        if "emb_key" in self.additional_states:
            x = [x_in[0]]
            params = [x_in[1]]
        else:
            x = x_in
        pnl = tf.zeros((tf.shape(x[0])[0],))
        self._prev_q = tf.zeros(
            (tf.shape(x[0])[0], len(self.hedge_instruments)), name="prev_q"
        )
        for i in range(self.timegrid.shape[0] - 2):
            t = (
                [self.timegrid[-1] - self.timegrid[i]]
                * tf.ones((tf.shape(x[0])[0], 1))
                / self.timegrid[-1]
            )
            inputs = [v[:, i] for v in x]
            if "emb_key" in self.additional_states:
                inputs.append(params)
            inputs.append(t)
            quantity = self.model(inputs, training=training)
            for j in range(len(self.hedge_instruments)):
                pnl += tf.math.multiply(
                    (self._prev_q[:, j] - quantity[:, j]), tf.squeeze(x[j][:, i])
                )
            self._prev_q = quantity
        for j in range(len(self.hedge_instruments)):
            pnl += self._prev_q[:, j] * tf.squeeze(x[j][:, -1])
        return pnl

    @tf.function
    def _compute_pnl_withconstains(self, x_in, training):
        if "emb_key" in self.additional_states:
            x = [x_in[0]]
            params = [x_in[1]]
        else:
            x = x_in
        pnl = tf.zeros((tf.shape(x[0])[0],))
        self._prev_q = tf.zeros(
            (tf.shape(x[0])[0], len(self.hedge_instruments)), name="prev_q"
        )
        for i in range(self.timegrid.shape[0] - 2):  # tensorflow loop?
            t = (
                [self.timegrid[-1] - self.timegrid[i]]
                * tf.ones((tf.shape(x[0])[0], 1))
                / self.timegrid[-1]
            )
            inputs = [v[:, i] for v in x]
            if "emb_key" in self.additional_states:
                inputs.append(params)
            inputs.append(t)
            quantity = self.model(inputs, training=training)
            for j in range(len(self.hedge_instruments)):
                key_to_check = self.hedge_instruments[j]
                if key_to_check in self.transaction_cost.keys():
                    tc = list(self.transaction_cost[key_to_check])
                    tf.cond(
                        tf.equal(len(self.timegrid), len(tc)),
                        lambda: None,
                        lambda: tc.extend([tc[-1]] * (len(self.timegrid) - len(tc))),
                    )
                else:
                    tc = [0] * len(self.timegrid)
                diff_q = self._prev_q[:, j] - quantity[:, j]
                xx = tf.squeeze(x[j][:, i])
                # Trading restriction based on threshold

                # tf.cond(
                #     tf.equal(self.threshold, 0.0),
                #     lambda: xx,
                #     lambda: tf.where(
                #         tf.greater(quantity[:, j], self.threshold),
                #         tf.zeros_like(xx),
                #         xx,
                #     ),
                # )
                # Cascading:
                # if self.cascading:
                #    tt = np.round(self.timegrid * 365.0, 0)
                #    if tt[i] % 7 != 0 and tt[i] >= 7:  # weekly
                #        xx = tf.zeros_like(xx)
                pnl += tf.where(
                    tf.greater(diff_q, 0),
                    tf.math.multiply(diff_q, tf.scalar_mul((1.0 - tc[0]), xx)),
                    tf.math.multiply(diff_q, tf.scalar_mul((1.0 + tc[0]), xx)),
                )
            self._prev_q = quantity

        for j in range(len(self.hedge_instruments)):
            key_to_check = self.hedge_instruments[j]
            if key_to_check in self.transaction_cost.keys():
                tc = list(self.transaction_cost[key_to_check])
                tf.cond(
                    tf.equal(len(self.timegrid), len(tc)),
                    lambda: None,
                    lambda: tc.extend([tc[-1]] * (len(self.timegrid) - len(tc))),
                )
            else:
                tc = [0] * len(self.timegrid)
            diff_q = self._prev_q[:, j] - quantity[:, j]
            xx = tf.squeeze(x[j][:, -1])
            # Trading restriction based on threshold
            # tf.cond(
            #    tf.equal(self.threshold, 0.0),
            #    lambda: xx,
            #    lambda: tf.where(
            #        tf.greater(quantity[:, j], self.threshold), tf.zeros_like(xx), xx
            #    ),
            # )
            pnl += tf.where(
                tf.greater(diff_q, 0),
                tf.math.multiply(self._prev_q[:, j], tf.scalar_mul((1.0 - tc[-1]), xx)),
                tf.math.multiply(self._prev_q[:, j], tf.scalar_mul((1.0 + tc[-1]), xx)),
            )
        return pnl

    def compute_delta(self, paths: Dict[str, np.ndarray], t: Union[int, float] = None):
        if t is None:
            result = np.zeros(
                (
                    self.timegrid.shape[0],
                    next(iter(paths.values())).shape[1],
                    len(self.hedge_instruments),
                )
            )
            for i in range(self.timegrid.shape[0]):
                result[i, :, :] = self.compute_delta(paths, i)
            return result
        if isinstance(t, int):
            inputs_ = self._create_inputs(paths)#, check_timegrid=True)
            inputs = []
            inputs.append(inputs_[0][:,t])
            inputs.append(inputs_[1]) #= [inputs_[i][:,t] for i in range(len(inputs_))]
            t = (self.timegrid[-1] - self.timegrid[t]) / self.timegrid[-1]
        else:
            inputs_ = self._create_inputs(paths, check_timegrid=True)
            inputs = [inputs_[i] for i in range(len(inputs_))]
        # for k,v in paths.items():
        inputs.append(np.full(inputs[0].shape, fill_value=t))
        return self.model.predict(inputs)

    def _compute_delta_path(self, paths: Dict[str, np.ndarray]):
        result = np.zeros((self.timegrid.shape[0], next(iter(paths.values())).shape[0]))
        for i in range(self.timegrid.shape[0]):
            result[i, :] = self.compute_delta(paths, i)
        return result

    def compute_pnl(self, paths: Dict[str, np.ndarray], payoff: np.ndarray):
        inputs = self._create_inputs(paths)
        return payoff + self.predict(inputs)  # -Z + d S


    @tf.function
    def custom_loss(self, y_true, y_pred):
        #
        if self._loss == "exponential_utility":
            return (
                tf.keras.backend.log(
                    tf.keras.backend.mean(
                        tf.keras.backend.exp(-self.regularization * (y_pred + y_true))
                    )
                )
                / self.regularization
            )
        elif self._loss == "expected_shortfall":
            es, _ = tf.nn.top_k(
                -(y_pred + y_true),
                tf.cast(self.regularization * y_true.shape[0], tf.int32),
            )
            return tf.reduce_mean(es)
        return -self.regularization * tf.keras.backend.mean(
            y_pred + y_true
        ) + tf.keras.backend.var(y_pred + y_true)

    def _create_inputs(
        self, paths: Dict[str, np.ndarray], check_timegrid: bool = True
    ) -> List[np.ndarray]:
        inputs = []
        if check_timegrid:
            for k in self.hedge_instruments:
                if paths[k].shape[1] != self.timegrid.shape[0]:
                    inputs.append(paths[k].transpose())
                else:
                    inputs.append(paths[k])
            for k in self.additional_states:

                if len(paths[k].shape) > 1 and (
                    paths[k].shape[1] != self.timegrid.shape[0]
                ):
                    inputs.append(paths[k].transpose())
                else:
                    inputs.append(paths[k])
        else:
            for k in self.hedge_instruments:
                inputs.append(paths[k])
            for k in self.additional_states:
                inputs.append(paths[k])
        return inputs


    def train(
        self,
        paths: Dict[str, np.ndarray],
        payoff: np.ndarray,
        lr_schedule,
        epochs: int,
        batch_size: int,
        tensorboard_log: str = None,
        verbose=0,
    ):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule
        )  # beta_1=0.9, beta_2=0.999)
        callbacks = []
        if tensorboard_log is not None:
            logdir = tensorboard_log  # os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                logdir, histogram_freq=0
            )
            callbacks.append(tensorboard_callback)
        self.compile(optimizer=optimizer, loss=self.custom_loss)
        inputs = self._create_inputs(paths)
        return self.fit(
            inputs,
            payoff,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            validation_split=0.1,
            validation_freq=5,
        )

    def save(self, folder):
        self.model.save(folder + "/delta_model")
        params = {}
        params["regularization"] = self.regularization
        params["timegrid"] = [x for x in self.timegrid]
        params["additional_states"] = self.additional_states
        params["hedge_instruments"] = self.hedge_instruments
        params["loss"] = self._loss
        params["transaction_cost"] = self.transaction_cost
        params["no_of_unique_model"] = self.no_of_unique_model
        # params["threshold"] = self.threshold
        with open(folder + "/params.json", "w") as f:
            json.dump(params, f)

    @staticmethod
    def load(folder: str):
        with open(folder + "/params.json", "r") as f:
            params = json.load(f)
        base_model = tf.keras.models.load_model(folder + "/delta_model")
        emb_layer = base_model.get_layer('Embedding')
        (w,) = emb_layer.get_weights()
        
        params["timegrid"] = np.array(params["timegrid"])
        params["additional_states"] = np.array(params["additional_states"])
        params["hedge_instruments"] = np.array(params["hedge_instruments"])
        params["no_of_unique_model"] = params["no_of_unique_model"]
        if not ("loss" in params.keys()):
            params["loss"] = "mean_variance"
        return DeepHedgeModelwEmbedding(depth=None, n_neurons=None, model=base_model, **params), (w,)
    

    def get_params(self)->np.ndarray:
        emb_layer = self.model.get_layer('Embedding')
        return emb_layer.get_weights()[0]
    
    def n_tasks(self) -> int:
        emb_layer = self.model.get_layer('Embedding')
        return emb_layer.input_dim  # -1
    

    def set_params(self, params: np.ndarray):
        new_params = self.get_params()
        if len(params.shape) == 1:
            new_params[-1,:] = params
        elif params.shape[0] == 1:
            new_params[-1,:] = params
        elif params.shape[0] == self.n_tasks():
            new_params[:-1,:] = params
        
        emb_layer = self.model.get_layer('Embedding')
        emb_layer.set_weights([new_params])

    
    def predict_single_pnl(self, paths: Dict[str, np.ndarray], payoff: np.ndarray):
        inputs = self._create_inputs(paths)
        return self.predict(inputs) + payoff 
    

    def fit_param(self, optimizer, callbacks, paths: Dict[str, np.ndarray], payoff: np.ndarray):


        #self.model.trainable = False
        #self.model.compile(optimizer=optimizer, loss=self.custom_loss)

        #emb_layer.trainable = True
        #self.compile(optimizer=optimizer, loss=self.custom_loss)
        #count = 0
        #for k,v in self.model._get_trainable_state().items():
        #    if count == 0:
        #        k.trainable = True
        #    if count == 4:
        #        k.trainable = True
        #        #self.compile(optimizer=optimizer, loss=self.custom_loss)
        #    else:
        #        k.trainable = False
        #    count = count + 1
        #    print(k.name,v)

        for layer in self.model.layers:
            if layer.name == 'Embedding':
                emb_layer = self.model.get_layer('Embedding')
                params = emb_layer.get_weights()
                print(params)
                emb_layer.trainable=True
            else:
                layer.trainable = False
            print(layer, layer.name, layer.trainable)
        self.compile(optimizer=optimizer, loss=self.custom_loss)
            #print(k.name,v)
            #print(k.get_weights())

        inputs = self._create_inputs(paths)
        return(self.fit(
            inputs,
            payoff,
            epochs=10,
            batch_size=500,
            callbacks=callbacks,
            verbose=1,
            validation_split=0.1,
            validation_freq=5))



    @staticmethod
    def train_task(model, paths: Dict[str, np.ndarray], payoff: np.ndarray, #X_test, task, 
                  initial_lr=0.005, epochs=100, 
                  batch_size=64, verbose=1, loss='mean_variance', 
                  decay_steps=16_000,decay_rate=0.95):

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=initial_lr,#1e-3,
                decay_steps=decay_steps,
                decay_rate=decay_rate, 
                staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        callbacks = []
        tensorboard_log = None
        if tensorboard_log is not None:
            logdir = tensorboard_log  # os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                logdir, histogram_freq=0
            )
            callbacks.append(tensorboard_callback)
        

        model.fit_param(optimizer=optimizer, callbacks=callbacks,paths=paths,payoff=payoff)
        #y_pred = model.predict_single_pnl(paths, payoff)
        return None#y_pred
    



    
    

