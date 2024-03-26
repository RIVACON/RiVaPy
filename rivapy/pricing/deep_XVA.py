from typing import List, Dict, Union
import json
import numpy as np
try:
    import tensorflow as tf
    try:
        tf.config.run_functions_eagerly(False)
    except:
        pass
except:
    import warnings
    warnings.warn('Tensorflow is not installed. You cannot use the PPA Deep Hedging Pricer!')


# Template for Deep XVA model: work in progress!!

class DeepXVAModel(tf.keras.Model):
    def __init__(self, instruments:List[str], 
                        timegrid: np.ndarray, 
                        depth: int,
                        n_neurons: int,
                        loss: str,
                        model: tf.keras.Model=None,
                        **kwargs):
        """ Class for Deep XVA Model.

        Args:
            instruments (List[str]): List of keys of the instruments used for valuation. The keys in this list are used in the dictionary of paths.
            timegrid (np.ndarray): The timegrid of the paths. The model input gets the current time to maturity as an additional input that is computed at gridpoint t as (timegrid[-1]-timegrid[t])/self.timegrid[-1].
            depth (int): If no model (neural network) is specified, the model is build as a fully connected neural network with this depth.
            n_neurons (int): If no model (neural network) is specified, the model is build as a fully connected neural network with this number of neurons per layer.
            loss (str): Determines whether mean variance or exponential utility are optimized.
            model (tf.keras.Model, optional): The model (neural network) used. If it is None, a fully connected neural network using the parameter depth and n_neurons. The network Input must equal the number of hedge instruments plus the number of additional states plus one input for the time to maturity. The output dimension must equal the number of hedge instruments. Defaults to None.
        """
        super().__init__(**kwargs)
        self.instruments = instruments
        if model is None:
            self.model = self._build_model(depth,n_neurons)
        else:
            self.model = model
        self.timegrid = timegrid
        self._prev_q = None
        self._forecast_ids = None
        self._loss = loss 

        
    def __call__(self, x, training=True):
        return self._compute_PV(x, training) 
    
    def _build_model(self, depth: int, nb_neurons: int):
        inputs= [tf.keras.Input(shape=(1,),name = ins) for ins in self.hedge_instruments]
        if self.additional_states is not None:
            for state in self.additional_states:
                inputs.append(tf.keras.Input(shape=(1,),name = state))
        inputs.append(tf.keras.Input(shape=(1,),name = "ttm"))
        fully_connected_Input = tf.keras.layers.concatenate(inputs)         
        values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(fully_connected_Input)       
        for _ in range(depth):
            values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)            
        value_out = tf.keras.layers.Dense(len(self.hedge_instruments), activation="linear",
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)
        model = tf.keras.Model(inputs=inputs, outputs = value_out)
        return model

    def _compute_PV(self, x, training):
        '''Function to compute PV: TODO'''
        return None#PV


    #def compute_delta(self, paths: Dict[str, np.ndarray],
    #                  t: Union[int, float]=None):
    #    if t is None:
    #        result = np.zeros((self.timegrid.shape[0], next(iter(paths.values())).shape[1], 
    #                          len(self.hedge_instruments)))
    #        for i in range(self.timegrid.shape[0]):
    #            result[i,:,:]=self.compute_delta(paths, i)
    #        return result
    #    if isinstance(t, int):
    #        inputs_ = self._create_inputs(paths, check_timegrid=True)
    #        inputs = [inputs_[i][:,t] for i in range(len(inputs_))]
    #        t = (self.timegrid[-1] - self.timegrid[t])/self.timegrid[-1]
    #    else:
    #        inputs_ = self._create_inputs(paths, check_timegrid=False)
    #        inputs = [inputs_[i] for i in range(len(inputs_))]
    #    #for k,v in paths.items():
    #    inputs.append(np.full(inputs[0].shape, fill_value=t))
    #    return self.model.predict(inputs)

    #def _compute_delta_path(self, paths: Dict[str, np.ndarray]):     
    #    result = np.zeros((self.timegrid.shape[0], next(iter(paths.values())).shape[0]))
    #    for i in range(self.timegrid.shape[0]):
    #        result[i,:]=self.compute_delta(paths, i)
    #    return result
    
    def compute_PV(self, 
                    paths: Dict[str, np.ndarray]):
        '''Function to compute PV'''
        inputs = self._create_inputs(paths)
        return self.predict(inputs)



    @tf.function
    def custom_loss(self, y_true, y_pred):
        '''loss function: TODO'''
        return - self.regularization*tf.keras.backend.mean(y_pred+y_true) + tf.keras.backend.var(y_pred+y_true)



    def _create_inputs(self, paths: Dict[str, np.ndarray], check_timegrid: bool=True)->List[np.ndarray]:
        inputs = []
        if check_timegrid:
            for k in self.instruments:
                if paths[k].shape[1] != self.timegrid.shape[0]:
                    inputs.append(paths[k].transpose())
                else:
                    inputs.append(paths[k])
            for k in self.additional_states:
                if paths[k].shape[1] != self.timegrid.shape[0]:
                    inputs.append(paths[k].transpose())
                else:
                    inputs.append(paths[k])
        else:
            for k in self.hedge_instruments:
                inputs.append(paths[k])
            for k in self.additional_states:
                inputs.append(paths[k])
        return inputs

    def train(self, paths: Dict[str,np.ndarray], 
            payoff: np.ndarray, 
            lr_schedule, 
            epochs: int, batch_size: int, 
            tensorboard_log:str=None,
            verbose=0):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #beta_1=0.9, beta_2=0.999)
        callbacks = []
        if tensorboard_log is not None:
            logdir = tensorboard_log#os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
            callbacks.append(tensorboard_callback)
        self.compile(optimizer=optimizer, loss=self.custom_loss)
        inputs = self._create_inputs(paths)
        return self.fit(inputs, payoff, epochs=epochs, 
                            batch_size=batch_size, callbacks=callbacks, 
                            verbose=verbose, validation_split=0.1, 
                            validation_freq=5)

    def save(self, folder):
        self.model.save(folder+'/delta_model')
        params = {}
        params['timegrid'] =[x for x in self.timegrid]
        params['instruments'] = self.instruments
        params['loss'] = self._loss
        with open(folder+'/params.json','w') as f:
            json.dump(params, f)

    @staticmethod
    def load(folder: str):
        with open(folder+'/params.json','r') as f:
            params = json.load(f)
        base_model = tf.keras.models.load_model(folder+'/delta_model')
        params['timegrid'] = np.array(params['timegrid'])
        if not ('loss' in params.keys()):
            params['loss']='mean_variance' 
        return DeepXVAModel(depth=None,n_neurons=None, model=base_model, **params)

    


