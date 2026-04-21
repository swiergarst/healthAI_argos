from typing import Any


from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

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
    client: AlgorithmClient, config_path: str
) -> Any:

    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]

    config = json.load(config_path)

    #TODO: add loading of initial weights

    for round in range (config['num_rounds']):


        # Define input parameters for a subtask
        info("Defining input parameters")
        input_ = {
            "method": "train_locally",
            "kwargs": {
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


        # average models
        # TODO: change this function such that it responds with the averaged model instead
        # alternatively, we could have the averaging as a subtask as well, and simply store the global model in the blob storage as well.
        # question then becomes how we could give access to that model to the clients
        # arguably this might not matter much, we have to transfer the global model to the clients either way 
        aggregated_model_path, model_name = avg.fed_average(node_model_path,iteration=variables['iteration'])

