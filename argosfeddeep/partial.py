import pandas as pd
import numpy as np

import argosfeddeep.models as mod
from vantage6.algorithm.tools.decorators import data
import tensorflow as tf
from .run_online import main, dice_bce
# from .local import *


@data(0)
def test_train_locally(config: dict, model_weights:list[np.ndarray, np.ndarray]):
    # reshape the dataframe into a 4d array (we're hardcoding for now, while it is test data anyway)
    # this part should also change when (mock) data of the right format is available
    # dset_train = prep_data(df_train, config['patch_shape'].copy())
    # dset_val = prep_data(df_val, config['patch_shape'].copy())
    # print(f'config: {config}')
    # print(f'dsets: {dset_train.take(1)}') 
    # print(dset_train.shape)
    main(model_weights, config)
    # print(dset_train, dset_val)    
    # print(labels_train, labels_val)

@data(0)  # TODO: change this to the right type of database once we know what that needs to be
def train_locally(config: dict, model_weights):

    # dset_train = prep_data(df_train, config['patch_shape'])
    # dset_val = prep_data(df_val, config['patch_shape'])
    # TODO: change run_online to take a dataset as input
    return main(model_weights, config)
 

# test function to debug weight averaging
@data(0)
def pass_weights_back(config: dict, model_weights:list):
    # Define model
    # Define optimizer with learning rate
    loss_function = dice_bce
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'],
                                                                  decay_steps=config['decay_steps'],
                                                                  decay_rate=config['decay_rate'],
                                                                  staircase=True)
    optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = mod.mod_resnet(config,
                        config['num_classes'],
                        optimizer=optimizer_function,
                        loss=loss_function)
    
    model_weights_conv = [np.array(model_weight) for model_weight in model_weights]
    print("setting model weights")
    print("model shape: ")
    print(f'{[layer.shape for layer in model_weights_conv]}')
    model.set_weights(model_weights_conv)

    print("returning model weights")
    returned_weights = [weight.tolist() for weight in model.get_weights()]
    # return the same weights just to keep the workflow the same
    return {
        "model weights" : returned_weights
    }