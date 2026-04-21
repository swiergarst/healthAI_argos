import os
from statistics import mode
from time import sleep
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import json
import datetime
import random
import nibabel as nib
from shutil import copyfile
from tensorflow.keras import losses
import argosfeddeep.utils as utl
import argosfeddeep.data_augmentation as aug
import argosfeddeep.models as mod
from typing import Any

data_path = "/home/swier/Documents/healthAI_argos/argos_layout/argos_layout/"
# data_path = '/mnt/data'

#with open (os.path.join(param_dir, params_file),'w') as fp:
  #  json.dump(params,fp)


def run_once(f):
    """
    Wrapper for functions that should only run once every run.

    Parameters
    ----------
    f : function
        Function to be ran.

    Returns
    -------
    wrapper : boolean
    """

    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice_score = self.add_weight(name='dsc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # smooth = 0.000001
        smooth = 1
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        score = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
        self.dice_score.assign(score)

    def result(self):
        return self.dice_score

    def reset_states(self):
        self.dice_score.assign(0.0)


def dice_loss(y_true, y_pred):
    smooth = 1
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    score = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return 1 - score


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """Soft dice loss calculation for arbitrary batch size, number of classes,
    and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = tf.ones(tf.shape(y_true))


    #ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones - y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = tf.math.reduce_sum(p0 * g0, axis=(0, 1, 2, 3))
    den = num + alpha * tf.math.reduce_sum(p0 * g1, axis=(0, 1, 2, 3)) + beta * tf.math.reduce_sum(p1 * g0, axis=(0, 1, 2, 3))

    # num = K.sum(p0*g0, (0,1,2,3))
    # den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    T = tf.math.reduce_sum(num/den)
    Ncl = tf.cast(tf.shape(y_true)[-1], dtype='float32')
    # T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    # Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def dice_score(y_true, y_pred, ignore_background=True, square=False):
    if ignore_background:
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
    y_pred_t = tf.where(tf.greater(y_pred, 0.15), 0, 1)
    y_pred_t = tf.dtypes.cast(y_pred_t, tf.float32)
    y_true = tf.dtypes.cast(y_true, tf.float32)
    axes = (0, 1, 2)
    eps = 1e-7
    num = (2 * tf.reduce_sum(y_true * y_pred, axis=axes) + eps)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes) + eps
    score = tf.reduce_mean(num / denom)

    return score


def dice_loss2(y_true, y_pred, ignore_background=False, square=False):
    if ignore_background:
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
    y_pred_t = tf.where(tf.greater(y_pred, 0.15), 0, 1)
    y_pred_t = tf.dtypes.cast(y_pred_t, tf.float32)
    y_true = tf.dtypes.cast(y_true, tf.float32)
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


def sort_slices(path, name):
    pos_dict = {}
    neg_dict = {}
    slice_dict = {}
    patients = os.listdir(path)
    for patient in patients:
    # patient = patients[0]

        patient_path = os.path.join(path, patient)
        ct_path = os.path.join(patient_path, 'CT')
        gt_path = os.path.join(patient_path, 'GT')
        gt_lung_path = os.path.join(gt_path, 'Lung')
        gt_gtv_path = os.path.join(gt_path, 'GTV')

        gt_pos = []
        gt_neg = []
        gt_slices = []
        for layer in range(0, len(os.listdir(ct_path))):

            # ct_patch = nib.load(os.path.join(ct_path, str(layer) + '_ct.nii.gz')).get_fdata()
            gt_patch_gtv = nib.load(os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz')).get_fdata()
            # pet_patch = nib.load(os.path.join(pet_path, str(layer) + '_pet.nii.gz')).get_fdata()
            if np.max(gt_patch_gtv) == 1:
                gt_slices.append(os.path.join(ct_path, str(layer) + '.nii.gz') + ',' + os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz') + ',' + os.path.join(gt_lung_path, str(layer) + '_lung.nii.gz') + ', ' + '1')


            else:
                gt_slices.append(os.path.join(ct_path, str(layer) + '.nii.gz') + ',' + os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz') + ',' + os.path.join(gt_lung_path, str(layer) + '_lung.nii.gz') + ', ' + '0')

        pos_dict[patient] = gt_pos
        neg_dict[patient] = gt_neg
        slice_dict[patient] = gt_slices

        with open(name, 'w') as fp:
            json.dump(slice_dict, fp)


def early_stopping(loss_list, min_delta=0.005, patience=20):
    """

    Parameters
    ----------
    loss_list : list
        List containing loss values for every evaluation.
    min_delta : float
        Float serving as minimum difference between loss values before early stopping is considered.
    patience : int
        Training will not be stopped before int(patience) number of evaluations have taken place.

    Returns
    -------

    """
    # TODO: Changed to list(loss_list)
    if len(list(loss_list)) // patience < 2:
        return False

    mean_previous = np.mean(loss_list[::-1][patience:2 * patience])
    mean_recent = np.mean(loss_list[::-1][:patience])
    delta_abs = np.abs(mean_recent - mean_previous)  # abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change

    if delta_abs < min_delta:
        print('Stopping early...')
        return True
    else:
        return False


@run_once
def _start_graph_tensorflow(log_dir:str):
    """
    Starts the tensorboard graph. Allows for the tracking of loss curves, accuracy and architecture visualization.
    log_dir : str
    Path to directory where updates should be stored.
    """
    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)


@run_once
def _end_graph_tensorflow(self):
    """

    Parameters
    ----------
    self : tf.writer
        train_summary_writer.
    Returns
    -------

    """
    with self.as_default():
        tf.summary.trace_export(name="graph", step=0)


def get_batch_full(ct_slices, params):
    # ct_path = os.path.join(patient_path, 'CT')
    # gt_path = os.path.join(patient_path, 'GT')
    # pet_path = os.path.join(patient_path, 'PT')


    ct = np.zeros(shape=[params['batch_size'], 512, 512, params['patch_shape'][2]])
    gt = np.zeros(shape=[params['batch_size'], 512, 512, 1])

    for layer in range(0, params['batch_size']):
        while True:
            random_case = random.choice(list(ct_slices))
            if len(ct_slices[random_case]) != 0:
                break
            else:
                print(str(random_case) + ' Length: ' + str(len(ct_slices[random_case])))



        rand_num = random.randint(0, 2)
        # print(str(random_case) + ' Length: ' + str(len(ct_slices[random_case])))
        if rand_num == 0:
            while True:
                random_layer = random.randint(0, len(ct_slices[random_case]) - 1 - (params['patch_shape'][2] // 2))
                selected_slice = ct_slices[random_case][random_layer]
                output = selected_slice.split(',')
                if int(output[-1]) == 1:

                    break
        else:
            random_layer = random.randint(0, len(ct_slices[random_case]) - 1 - (params['patch_shape'][2] // 2))
            selected_slice = ct_slices[random_case][random_layer]
            output = selected_slice.split(',')

        min_layer = random_layer - params['patch_shape'][2] // 2

        gt_patch = nib.load(output[1]).get_fdata()

        ct_patch = np.zeros([params['patch_shape'][0],
                             params['patch_shape'][1],
                             params['patch_shape'][2]])

        for z in range(0, params['patch_shape'][-1]):
            selected_slice = ct_slices[random_case][min_layer + z]
            output = selected_slice.split(',')
            ct_patch[:, :, z] = nib.load(output[0]).get_fdata().reshape(512,512)


        if random.randint(0, 1) == 1:
            num_augments = np.random.randint(1, params['number_of_augmentations'] + 1)
            ct_patch, gt_patch = aug.apply_augmentations(ct_patch,
                                                                       gt_patch,
                                                                       num_augments)

        ct[layer, :, :, :] = ct_patch
        gt[layer, :, :, 0] = gt_patch.reshape(512, 512)
    gt = tf.one_hot(np.uint8(np.squeeze(gt, axis=-1)), params['num_classes'])
    return ct, gt


def main(model_weights: Any, params: dict):
    @tf.function
    def train_on_batch(im_src, gt_src):
        """
        Manages and updates parameters for training.
        Parameters
        ----------
        im_src : np.ndarray
        gt_src : np.ndarray
        pet_src : np.ndarray

        Returns
        -------

        """
        with tf.GradientTape() as tape:
            predictions = model(inputs=[im_src], training=True)
            regularization_loss = tf.math.add_n(model.losses)
            #print()
            #print(np.shape(predictions))
            #print()
            #print(np.shape(gt_src))
            loss_value = loss_function(gt_src, predictions)
            total_loss = regularization_loss + loss_value

        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer_function.apply_gradients(zip(grads, model.trainable_weights))
        train_loss(total_loss)
        return predictions

    @tf.function
    def validate_on_batch(im_src, gt_src):
        """
        Manages validation.

        Parameters
        ----------
        im_src : np.ndarray
        gt_src : np.ndarray
        pet_src : np.ndarray

        Returns
        -------

        """
        predictions = model(inputs=[im_src], training=False)
        regularization_loss = tf.math.add_n(model.losses)
        loss_value = loss_function(gt_src, predictions)
        total_loss = regularization_loss + loss_value
        validation_loss(total_loss)
        return predictions

    
    # param_path = os.path.join(data_path , 'assets','params.json')
    # params = utl.Params(param_path)

    sort_slices(os.path.join(data_path,'Train/'),'slices_training_modified.json')

    sort_slices(os.path.join(data_path,'Validation/'),'slices_validation_modified.json')
    # Define loss function
    loss_list = []
    # loss_function = losses.CategoricalCrossentropy()
    # loss_function = dice_loss2
    loss_function = dice_bce

    # Define optimizer with learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params['learning_rate'],
                                                                  decay_steps=params['decay_steps'],
                                                                  decay_rate=params['decay_rate'],
                                                                  staircase=True)
    optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer_function = tf.keras.optimizers.Adam(params.dict['learning_rate'])

    # Define model
    model = mod.mod_resnet(params,
                        params['num_classes'],
                        optimizer=optimizer_function,
                        loss=loss_function)

    # if averaged_model_path is not None:
    # model.load_weights(averaged_model_path)
    # weights = model.get_weights()
    if model_weights is not None:
        print('Loading previous weight from: central weights')
        model.set_weights(model_weights)

    # print(model.summary)
    # Define evaluation metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = dice_score
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    validation_accuracy = dice_score

    # Create variables for various paths used for storing training information
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(os.path.join(os.getcwd(),'logs','gradient_tape','train')):
        os.makedirs(os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','train'))
    train_log_dir = os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','train')
    if not os.path.exists(os.path.join(os.getcwd(),'logs','gradient_tape','val')):
        os.makedirs(os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','val'))
    val_log_dir = os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','val')
    if not os.path.exists(os.path.join(os.getcwd(),'logs','gradient_tape','saved_models')):
        os.makedirs(os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','saved_models'))
    saved_model_path = os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','saved_models')
    if not os.path.exists(os.path.join(os.getcwd(),'logs','gradient_tape','saved_weights')):
        os.makedirs(os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','saved_weights'))
    saved_weights_path = os.path.join(os.getcwd(),'logs',current_time,'gradient_tape','saved_weights')

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)


    # Load training and validation data
    train_slices = utl.read_slices(os.path.join(data_path,'assets','slices_training_modified.json'))
    validation_slices = utl.read_slices(os.path.join(data_path,'assets','slices_validation_modified.json'))
    

    iteration_number = list()
    train_loss_list = list()
    train_dice_list = list()
    validation_loss_list = list()
    validation_dice_list = list()


    # Start training loop
    for iteration_deep in range(0, params['num_steps'] + 1):
        # print(iteration_deep)
        _start_graph_tensorflow(train_log_dir)
        # dset_train = dset_train.shuffle()
        # batch = dset_train.shuffle(100).take(params['batch_size'])
        # ct_batch = np.array([sample[0] for sample in batch])
        # gt_batch = np.array([sample[1] for sample in batch])
        # print(f'ct_batch: {ct_batch}')
        # print(f'gt_batch: {gt_batch}') 
        ct_batch, gt_batch = get_batch_full(train_slices, params)

        train_pred = train_on_batch(ct_batch, gt_batch)

        _end_graph_tensorflow(train_summary_writer)

        # Evaluation step during training.
        if iteration_deep % params['train_eval_step'] == 0:
            # Write training information to training log
            with train_summary_writer.as_default():
                # train_dice = train_accuracy(gt_batch, train_pred).numpy()
                train_dice = train_accuracy(gt_batch, train_pred)

                tf.summary.scalar('loss', train_loss.result(), step=iteration_deep)
                tf.summary.scalar('accuracy', train_dice, step=iteration_deep)


            template = 'Iteration {}, Loss: {:.5}, Dice: {:.5}'
            print(template.format(iteration_deep + 1,
                                  train_loss.result(),
                                  train_dice))

        # Evaluation step for validation.
        if iteration_deep % params['val_eval_step'] == 0:
            # ct_batch_val, gt_batch_val = dset_val
            ct_batch_val, gt_batch_val = get_batch_full(validation_slices, params)
            val_pred = validate_on_batch(ct_batch_val, gt_batch_val)

            # Write validation information to log
            with val_summary_writer.as_default():
                # validation_dice = validation_accuracy(gt_batch_val, val_pred).numpy()
                validation_dice = validation_accuracy(gt_batch_val, val_pred)
                tf.summary.scalar('loss', validation_loss.result(), step=iteration_deep)
                tf.summary.scalar('accuracy', validation_dice, step=iteration_deep)
                loss_list.append(validation_loss.result())
            template = 'Iteration {}, Validation Loss: {:.5}, Validation Dice: {:.5}'
            print(template.format(iteration_deep + 1,
                                  validation_loss.result(),
                                  validation_dice))

            iteration_number.append(iteration_deep)
            train_loss_list.append(train_loss.result().numpy())
            train_dice_list.append(train_dice.numpy())
            validation_loss_list.append(validation_loss.result().numpy())
            validation_dice_list.append(validation_dice.numpy())

            # Earling stopping when loss in the past 'patience' train_eval_steps
            # is smaller than 'min_delta'. Breaks loop.
            # early_stop = early_stopping(loss_list, min_delta=0.01, patience=10)
            # if early_stop:
            #     print("Early stopping signal received at iteration = %d/%d" % (iteration, params.dict['num_steps']))
            #     print("Terminating training ")
            #     model.save(os.path.join(saved_model_path,
            #                             'model_' + str(iteration)))
            #     model.save_weights(os.path.join(saved_weights_path,
            #                                     'model_weights' + str(iteration) + '.h5'))
            #     break

        # Save the model at predefined step numbers.
        if iteration_deep % params['save_model_step'] == 0:
            model.save(f'{saved_model_path}model_{iteration_deep}.keras')
            model.save_weights(f'{saved_weights_path}model_weights_{iteration_deep}.weights.h5')
            # model.save_weights(os.path.join(saved_weights_path,
            #                                 'model_weights' + str(iteration_deep) + '.h5'))

            trained_model_path = os.path.join(saved_weights_path,'model_weights' + str(iteration_deep) + '.h5')
    
    model_metrics = {'node_iteration': iteration_number,
                     'training_loss': train_loss_list,
                     'training_dice':train_dice_list,
                     'validation_loss':validation_loss_list,
                     'validation_dice':validation_dice_list}


    return trained_model_path, model_metrics


def run_deep_algo(averaged_model_path,org_id,iteration):

    import shutil

    if not os.path.exists(os.path.join(os.getcwd(),'trained_model')):
        os.mkdir(os.path.join(os.getcwd(),'trained_model'))
    trained_model_directory = os.path.join(os.getcwd(),'trained_model')

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Running on CPU. Please install GPU version of TF")
    model_path, model_metrics = main(averaged_model_path)
    new_model_path = os.path.join(trained_model_directory,str(org_id)+'_'+str(iteration)+'_'+'node.h5')

    shutil.move(model_path,new_model_path)

    return new_model_path, model_metrics

#path, metrics = run_deep_algo(os.path.join(os.getcwd(),('initial_weight.h5')),2,1)

#print(path, metrics)


