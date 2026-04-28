from typing import Any


from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from argosfeddeep.models import mod_resnet
from argosfeddeep.run_online import dice_bce
from argosfeddeep.average import compute_average_weight
import tensorflow as tf
import json

# @algorithm_client
# def test_central(
#     client: AlgorithmClient, config_path: str
# ) -> Any:
#     organizations = client.organization.list()
#     org_ids = [organization.get("id") for organization in organizations]

#     config = json.load(config_path)
    

@algorithm_client
def central(
    client: AlgorithmClient, config: dict
) -> Any:

    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]

    # config = json.load(config_path)

    #TODO: add loading of initial weights
    # Define model so we can pull initial weights off of it
    # Define optimizer with learning rate (we need this to define a model)
    loss_function = dice_bce
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'],
                                                                    decay_steps=config['decay_steps'],
                                                                    decay_rate=config['decay_rate'],
                                                                    staircase=True)
    optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    init_model = mod_resnet(config,
                    config['num_classes'],
                    optimizer=optimizer_function,
                    loss=loss_function)

    init_weights = init_model.get_weights()
    weights_list = [init_weight.tolist() for init_weight in init_weights]

    for round in range (config['num_rounds']):
        print(f"round {round + 1} of {config['num_rounds']}")

        # Define input parameters for a subtask
        info("Defining input parameters")
        input_ = {
            "method": "train_locally",
            "kwargs": {
                "config" : config,
                "model_weights" : weights_list
            }
        }

        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=org_ids,
            name=f"training round {round}",
            description=""
        )

        # wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        # TODO: consider adding error checking on responses
        local_weights_list = [result['model weights'] for result in results]

        # average models
        # TODO: change this function such that it responds with the averaged model instead
        # alternatively, we could have the averaging as a subtask as well, and simply store the global model in the blob storage as well.
        # question then becomes how we could give access to that model to the clients
        # arguably this might not matter much, we have to transfer the global model to the clients either way 
        averaged_weights = compute_average_weight(local_weights_list)
        # aggregated_model_path, model_name = avg.fed_average(node_model_path,iteration=variables['iteration'])
        weights_list  = [weight.tolist() for weight in averaged_weights ]
    return True
