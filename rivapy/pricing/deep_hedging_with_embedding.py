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
        "Tensorflow is not installed. You cannot use the Deep Hedging Pricer!"
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
        embedding_size: int = 1,
        no_of_portfolios: int = 1, 
        embedding_size_port: int = 1,
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
                self.embedding_size = embedding_size#32
                self._embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.no_of_unique_model+1,
                    output_dim=self.embedding_size,
                    input_length=1,
                    name="Embedding",
                )
            if "port_key" in self.additional_states:
                self.no_of_portfolios = no_of_portfolios
                self.embedding_size_port = embedding_size_port
                self._embedding_layer_port = tf.keras.layers.Embedding(
                    input_dim=self.no_of_portfolios+1,
                    output_dim=self.embedding_size_port,
                    input_length=1,
                    name="Embedding_port",
                )
            self.model = self._build_model(depth, n_neurons)
        else:
            self.model = model
            self._embedding_layer = self.model.get_layer('Embedding')
            self._embedding_layer_port = self.model.get_layer('Embedding_port')
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
            return self._compute_pnl_withconstraints(x, training)  # + self.price
    

    def _build_model(self, depth: int, nb_neurons: int):
        inputs = [
            tf.keras.Input(shape=(1,), name=ins) for ins in self.hedge_instruments
        ]
        inputs.append(tf.keras.Input(shape=(1,), name="ttm"))
        fully_connected_Input1 = tf.keras.layers.concatenate(inputs)
        if self.additional_states is not None:
            for state in self.additional_states:
                if state not in ["emb_key","port_key"]:
                    inp_cat_data = tf.keras.layers.Input(shape=(1,), name=state)
                    inputs.append(inp_cat_data)
                elif state == "emb_key":
                    inp_cat_data = tf.keras.layers.Input(shape=(1,), name=state)
                    inputs.append(inp_cat_data)
                    emb = self._embedding_layer(inp_cat_data)
                    flatten = tf.keras.layers.Flatten()(emb)
                elif state == "port_key":
                    inp_cat2_data = tf.keras.layers.Input(shape=(1,), name=state)
                    inputs.append(inp_cat2_data)
                    emb_port = self._embedding_layer_port(inp_cat2_data)
                    flatten2 = tf.keras.layers.Flatten()(emb_port)



        # if "emb_key" in self.additional_states:
        #     inp_cat_data = tf.keras.layers.Input(shape=(1,))
        #     inputs.append(inp_cat_data)
        #     emb = self._embedding_layer(inp_cat_data)
        #     flatten = tf.keras.layers.Flatten()(emb)
        # if "port_key" in self.additional_states:
        #     inp_cat2_data = tf.keras.layers.Input(shape=(1,))
        #     inputs.append(inp_cat2_data)
        #     emb_port = self._embedding_layer_port(inp_cat2_data)
        #     flatten2 = tf.keras.layers.Flatten()(emb_port)
        if (("emb_key" in self.additional_states) and ("port_key" in self.additional_states)):   
            fully_connected_Input = tf.keras.layers.concatenate(
                    [fully_connected_Input1, flatten,flatten2]
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
        if (("emb_key" in self.additional_states) and ("port_key" in self.additional_states)):   
            x = x_in[:-2]
            params = [x_in[-2]]
            params_port = [x_in[-1]]
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
                inputs.append(params)
                inputs.append(params_port)
            quantity = self.model(inputs,training)
            for j in range(len(self.hedge_instruments)):
                pnl += tf.math.multiply(
                    (self._prev_q[:, j] - quantity[:, j]), tf.squeeze(x[j][:, i])
                )
            self._prev_q = quantity
        for j in range(len(self.hedge_instruments)):
            pnl += self._prev_q[:, j] * tf.squeeze(x[j][:, -1])
        return pnl


    @tf.function
    def _compute_pnl_withconstraints(self, x_in, training):
        #if (("emb_key" in self.additional_states) and ("port_key" in self.additional_states)):   
        #    x = x_in[:-2]

        #    params = [x_in[-2]]
        #    params_port = [x_in[-1]]
        if self.additional_states is not None:
            length = len(self.additional_states)
            x = x_in[:-length]
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
            inputs.append(t)
            if self.additional_states is not None:
                for ii in range(length,0,-1):
                    inputs.append([x_in[-ii]])
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
            xx = tf.squeeze(x[j][:, -1])[x_in[-2]]
            pnl += tf.where(
                tf.greater(diff_q, 0),
                tf.math.multiply(self._prev_q[:, j], tf.scalar_mul((1.0 - tc[-1]), xx)),
                tf.math.multiply(self._prev_q[:, j], tf.scalar_mul((1.0 + tc[-1]), xx)),
            )
        return pnl

    def compute_delta(self, paths: Dict[str, np.ndarray], t: Union[int, float] = None,emb: Union[int, float] = None, emb_port: Union[int, float] = None):
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
            #inputs = []
            #inputs.append(inputs_[0][:,t])
            inputs = [inputs_[i][:,t] for i in range(len(inputs_)-2)]
            #inputs.append(inputs_[1]) #= [inputs_[i][:,t] for i in range(len(inputs_))]
            t = (self.timegrid[-1] - self.timegrid[t]) / self.timegrid[-1]
        else:
            inputs_ = self._create_inputs(paths, check_timegrid=True)
            inputs = [inputs_[i] for i in range(len(inputs_))]
        # for k,v in paths.items():
        inputs.append(np.full(inputs[0].shape, fill_value=t))
        inputs.append(np.full(inputs_[-2].shape, fill_value=emb))
        inputs.append(np.full(inputs_[-1].shape, fill_value=emb_port))
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
        params["embedding_size"] = self.embedding_size
        params["no_of_portfolios"] = self.no_of_portfolios
        params["embedding_size_port"] = self.embedding_size_port
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
        emb_layer2 = base_model.get_layer('Embedding_port')
        (w2,) = emb_layer2.get_weights()
        
        params["timegrid"] = np.array(params["timegrid"])
        params["additional_states"] = np.array(params["additional_states"])
        params["hedge_instruments"] = np.array(params["hedge_instruments"])
        params["no_of_unique_model"] = params["no_of_unique_model"]
        params["embedding_size"] = params["embedding_size"]
        params["no_of_portfolios"] = params["no_of_portfolios"]
        params["embedding_size_port"] = params["embedding_size_port"]
        if not ("loss" in params.keys()):
            params["loss"] = "mean_variance"
        return DeepHedgeModelwEmbedding(depth=None, n_neurons=None, model=base_model, **params), (w,), (w2,)
    

    def get_params(self)->np.ndarray:
        emb_layer = self.model.get_layer('Embedding')
        return emb_layer.get_weights()#[0]
    
    def n_tasks(self) -> int:
        emb_layer = self.model.get_layer('Embedding')
        return emb_layer.input_dim  # -1
    

    def set_params(self, params: np.ndarray):
        emb_layer = self.model.get_layer('Embedding')
        emb_layer.set_weights([params])


    def fit_param(self, optimizer, callbacks, paths: Dict[str, np.ndarray], payoff: np.ndarray,emb: int, emb_port: int):


        for layer in self.model.layers:
            if layer.name == 'Embedding':
                emb_layer = self.model.get_layer('Embedding')
                params = emb_layer.get_weights()
                params[0][emb,:] =  params[0][:-1,:].mean(axis=0)
                emb_layer.set_weights(params)
                emb_layer.trainable=True
            elif layer.name == 'Embedding_port':
                emb_layer_port = self.model.get_layer('Embedding_port')
                params2 = emb_layer_port.get_weights()
                params2[0][emb_port,:] =  params2[0][:-1,:].mean(axis=0)
                emb_layer_port.set_weights(params2)
                emb_layer_port.trainable=True
            else:
                layer.trainable = False
            print(layer, layer.name, layer.trainable)
        self.compile(optimizer=optimizer, loss=self.custom_loss)
        inputs = self._create_inputs(paths)
        self.fit(
             inputs,
             payoff,
             epochs=20,
             batch_size=5,
             callbacks=callbacks,
             verbose=1,
             validation_split=0.1,
             validation_freq=5)
        for layer in self.model.layers:
                layer.trainable = True
        self.compile(optimizer=optimizer, loss=self.custom_loss)



    @staticmethod
    def train_task(model, paths: Dict[str, np.ndarray], payoff: np.ndarray, paths_test: Dict[str, np.ndarray], payoff_test, 
                   emb: int, emb_port: int, t: int,
                  initial_lr=0.0001, 
                  decay_steps=200,decay_rate=0.95):

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
        

        model.fit_param(optimizer=optimizer, callbacks=callbacks,paths=paths,payoff=payoff,emb=emb,emb_port=emb_port)
        y_pred = model.compute_pnl(paths, payoff)
        y_test = model.compute_pnl(paths_test,payoff_test)
        y_delta = model.compute_delta(paths_test, t=t,emb=emb,emb_port=emb_port)
        inputs = model._create_inputs(paths_test)
        y_loss = model.evaluate(inputs, payoff_test)
        return y_pred, y_test,y_delta, y_loss
    



    
    

