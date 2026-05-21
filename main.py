"""
main entry point for running the ARGOS usecase on Vantage6
"""

from argosfeddeep.run_online import dice_bce
from argosfeddeep.models import mod_resnet
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path
import json
import tensorflow as tf



def main(client, input: dict, organizations: list):
    # Run the central method on 1 node and get the results
    task = client.task.create(
        input_= input,
        organizations=organizations,
    )
    results = client.wait_for_results(task.get("id"))
    # print(results)
    return results




if __name__ == "__main__":
    # TODO: make sure we run the central task when this file is ran
    current_path = Path(__file__).parent
    data_path = f'{current_path}/argos_layout/argos_layout/'

    # read in config file
    with open(f'{current_path}/config.json', "r") as fp:
        config = json.load(fp)

    # setup vantage stuff
    task_input={
        "method":"central",
        "kwargs": {
            "config": config,
            # "model_weights" : weights_list
        }
    }

    ## Mock client
    # TODO: replace this with 'actual' v6 client
    client = MockAlgorithmClient(
        datasets=[[],[]],
        module="argosfeddeep"
    )

    organizations = client.organization.list()
    org_ids = [organizations[0]['id']] # only send central task to one org. TODO: figure out if this needs to be an external/mock org later on (for safety)


    main(client, task_input, org_ids)
