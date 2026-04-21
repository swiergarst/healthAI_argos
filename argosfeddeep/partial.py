import pandas as pd
import numpy as np

from vantage6.algorithm.tools.decorators import data
from tensorflow.data import Dataset
import tensorflow as tf
from .run_online import main
# from .local import *


# @data(2)
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

@data(2)  # TODO: change this to the right type of database once we know what that needs to be
def train_locally(df_train: pd.DataFrame, df_val: pd.DataFrame, config: dict, model_weights):

    dset_train = prep_data(df_train, config['patch_shape'])
    dset_val = prep_data(df_val, config['patch_shape'])
    # TODO: change run_online to take a dataset as input
    main(dset_train, dset_val, model_weights, config)
    
def test_forward_pass(df: pd.DataFrame, config: dict, model:tf.keras.Model):
    print(f'data shape before prep data: {df.shape}')
    
    dset = prep_data(df, config['patch_shape'])

    batch = dset.shuffle(100).take(config['batch_size'])
    ct_batch = np.array([sample[0] for sample in batch])
    gt_batch = np.array([sample[1] for sample in batch])
    predictions = model(inputs=[ct_batch], training=True)
    return predictions



def prep_data(df: pd.DataFrame, data_shape: list):
    labels= df['labels'].values
    data_raw = df.drop(columns = ['labels']).values
    
    data_shape.insert(0, -1)


    data_raw = data_raw.reshape(data_shape)
    print(f'data shape: {data_raw.shape}')
    # data_raw_val = df_val.drop(columns = ['labels']).values
    # labels_val = df_val_
    features = Dataset.from_tensor_slices(data_raw)
    labels = Dataset.from_tensor_slices(labels)
    
    dset = Dataset.zip((features, labels))
    # dset = Dataset.from_tensors((data_raw, labels))

    # dset_val = Dataset.from_tensors(data_raw_val.reshape((config['patch_shape'], -1)))
    return dset
