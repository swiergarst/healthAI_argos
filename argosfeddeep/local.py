import tensorflow as tf


@tf.function
def train_on_batch(model, im_src, gt_src):
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
    loss_function = 
    optimizer_function = 
    train_loss = 

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
def validate_on_batch(model, im_src, gt_src):
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