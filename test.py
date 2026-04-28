"""
local testing using MockAlgorithmClient
"""
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path
import json
from main import main as run_vantage
import tensorflow as tf
from argosfeddeep.run_online import dice_bce
from argosfeddeep.models import mod_resnet
from argosfeddeep.average import compute_average_weight
import numpy as np
import pandas as pd
# get path of current directory
current_path = Path(__file__).parent


data_path = f'{current_path}/argos_layout/argos_layout/'



# read in config file
with open(f'{current_path}/config.json', "r") as fp:
    config = json.load(fp)

## Mock client
client = MockAlgorithmClient(
    datasets=[[],[]],
        # Data for first organization
        # [{
        #     "database": current_path / "mock_data_train.csv",
        #     "db_type": "csv",
        #     "input_data": {}
        # },
        # {
        #     "database": current_path / "mock_data_val.csv",
        #     "db_type" : "csv",
        #     "input_data" : {}
        # }],
        # Data for second organization
    #     [{
    #         "database": current_path / "mock_data_train.csv",
    #         "db_type": "csv",
    #         "input_data": {}
    #     },
    #     {
    #         "database": current_path / "mock_data_val.csv",
    #         "db_type" : "csv",
    #         "input_data" : {}}]
    # ],
    module="argosfeddeep"
)

# Define optimizer with learning rate (we need this to define a model)
loss_function = dice_bce
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'],
                                                                decay_steps=config['decay_steps'],
                                                                decay_rate=config['decay_rate'],
                                                                staircase=True)
optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Define model so we can pull initial weights off of it
init_model = mod_resnet(config,
                config['num_classes'],
                optimizer=optimizer_function,
                loss=loss_function)

init_weights = init_model.get_weights()
weights_list = [init_weight.tolist() for init_weight in init_weights]

# df_train = pd.read_csv(current_path / "mock_data_train.csv")
# df_val = pd.read_csv(current_path / "mock_data_val.csv")

# test_train_locally(config, init_weights)
# pred = test_forward_pass(df_train, config, init_model)
# print(pred.shape)
# df_val = pd.read_csv(current_path / "mock_data_val.csv")

# initial_weights = tf.keras.models.load_model(f'{current_path}/initial_weight.h5')
# print(model.get_weights())
# for i in range(2):
#     print(f'round {i+1} of 2')
task_input={
    "method":"central",
    "kwargs": {
        "config": config,
        # "model_weights" : weights_list
    }
}


# list mock organizations
organizations = client.organization.list()
# print(organizations)
# org_ids = [organization["id"] for organization in organizations]
org_ids = [organizations[0]['id']] # only send central task to one org. TODO: figure out if this needs to be an external/mock org later on (for safety)

results = run_vantage(client, task_input, org_ids)

    # # wait for node to return results of the subtask.
    # # info("Waiting for results")
    # # results = client.wait_for_results(task_id=task.get("id"))
    # # info("Results obtained!")

    # # TODO: consider adding error checking on responses
    # local_weights_list = [result['model weights'] for result in results]
    # averaged_weights = compute_average_weight(local_weights_list)
    # weights_list = [weight.tolist() for weight in averaged_weights ]



