"""
main entry point for running the ARGOS usecase on Vantage6
"""

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pathlib import Path
import json



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
    main()
