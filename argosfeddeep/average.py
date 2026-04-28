import os
import numpy as np
import tensorflow as tf
import argosfeddeep.models as mod
import argosfeddeep.app as ap
import argosfeddeep.utils as utl
import time

data_path = '/mnt/data'

def dice_loss2(y_true, y_pred, ignore_background=True, square=False):
    if ignore_background:
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
    axes = (0, 1, 2)
    eps = 1e-7
    num = (2 * tf.reduce_sum(y_true * y_pred, axis=axes) + eps)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes) + eps
    score = tf.reduce_mean(num / denom)
    return 1 - score


def bce(y_true, y_pred):
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return binary_cross_entropy(y_true, y_pred)


def dice_bce(y_true, y_pred):
    d_l = dice_loss2(y_true, y_pred)
    bce_l = bce(y_true, y_pred)
    return d_l + bce_l

def construct_model():
    param_path = os.path.join(data_path ,'assets','params.json')
    params = utl.Params(param_path)
    
    # Define loss function
    loss_function = dice_bce
    
    # Define optimizer with learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params.dict['learning_rate'],
                                                                  decay_steps=params.dict['decay_steps'],
                                                                  decay_rate=params.dict['decay_rate'],
                                                                  staircase=True)
    optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer_function = tf.keras.optimizers.Adam(params['learning_rate'])
    
    # Define model
    model = mod.mod_resnet(params.dict,
                        params.dict['num_classes'],
                        optimizer=optimizer_function,
                        loss=loss_function)
    return model


def extract_weight(model, weight_path):
    # Load_weights only loads the weights into the defined modeled. It returns
    # a None object. Extract the numpy values using models.get_weights()
    model.load_weights(weight_path)
    weights = model.get_weights()
    return weights


def save_averaged_weights(model, average_weights, save_path, iteration):
    model.set_weights(average_weights)
    model_name = os.path.join(save_path, 'average_weight_iteration_'+str(iteration)+'.h5')
    model.save_weights(model_name)
    return model_name


# def compute_average_weight(model, weights_path):
def compute_average_weight(weights_list):

    # weights_list = []
    average_weights = list()
    
    # weights_files = os.listdir(weights_path)
    
    # for weights_file in weights_files:
    #     weights_list.append(extract_weight(model, os.path.join(weights_path, weights_file)))

    for weights_list_tuple in zip(*weights_list):
        # print(f'local layer shape: {weights_list_tuple.shape}')
        average_weights.append(np.mean(weights_list_tuple, axis=0))
    
    return average_weights


def fed_average(weights_path,iteration):
    
    save_path = ap.app.config['DOWNLOAD_FOLDER']
    model = construct_model()
    average_weight = compute_average_weight(model, weights_path)
    model_name = save_averaged_weights(model, average_weight, save_path, iteration)
    return save_path, model_name
