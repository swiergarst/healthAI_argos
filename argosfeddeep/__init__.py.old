from selectors import EpollSelector
import time
import json
import sys
from types import SimpleNamespace
from urllib import response
from flask import jsonify
import argosfeddeep.helper as dh
import argosfeddeep.org_id as do
import argosfeddeep.run_online as run
import argosfeddeep.database as db
import argosfeddeep.average as avg
import argosfeddeep.app as ap
import argosfeddeep.params as prm
import requests
import h5py
import os
import subprocess
import logging



# loggers
info = lambda msg: sys.stdout.write("info > " + msg + "\n")
warn = lambda msg: sys.stdout.write("warn > " + msg + "\n")

database_argos = r"/mnt/data/argos.db"

def master(client, data, org_ids, iteration_start, max_iteration):

    '''
    Retrieve token from master node . This token is essential for sending and receiving tasks from the data nodes
    Define variables required for sending as task input
    Retrieve organization ids from input parameter from the researcher   
    
    '''
    #Get jwt token from master node 
    token = dh.get_token()
    info(f"API token '{token}'")

    prm.set_params()
    info("params written to folder location")

    #start api for model upload download in the background
    subprocess.Popen(['python','/app/argosfeddeep/master_api.py','&'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    #Define variables
    variables = {"data_path" : "/mnt/data",
                "iteration": iteration_start,
                "max_iteration" : max_iteration
                }
    
    ids = org_ids

    conn = db.create_connection(database_argos)
    #flush_database(conn)
    #flush_all_folders()


    '''
    Iterative process for sending and receiving tasks and computing results
    '''    
    while variables['iteration']<=variables['max_iteration']:

        #check for iteration number , if iteration number more than 0, check for database entries, the initial model . if initial model is not set, 

        #Define Input for the subtask
        info("Defining input parameters")
        input_ = {
            "method": "deepnode",
            "args": [],
            "kwargs": {
                "token": token,
                "iteration": variables['iteration']
            }
        }


        info("Creating node tasks")
        task = client.create_new_task(
            input_,
            organization_ids=ids
        )

        info("Waiting for results")
        task_id = task.get("id")
        task = client.get_task(task_id)
        print(task)
        while not task.get("complete"):
            task = client.get_task(task_id)
            info("Waiting for results")
            time.sleep(15)

        results_master = client.get_results(task_id=task.get("id"))
        organization_ids = []
        for results in results_master:
            if results['flag']==0:
                organization_ids.append(results['Org id'])

        if organization_ids:
            message_from_master={'Organization ids uncompleted task':organization_ids}
            break

        #check database if all nodes returned back their updated model
        # create database connection
        
        while db.check_database_entries(conn,variables['iteration']) !=len(org_ids):
            time.sleep(5)
        info("All database entries received")
        info("Received all results")
        node_model_path = os.path.join(ap.app.config['UPLOAD_FOLDER'],str(variables['iteration']))
        aggregated_model_path, model_name = avg.fed_average(node_model_path,iteration=variables['iteration'])
        aggregated_model_path_name = os.path.join(aggregated_model_path,model_name)
        nodeType = "master"
        iteration=variables['iteration']
        value=(nodeType,iteration,aggregated_model_path_name)
        db.insert_into_table_aggregate(conn,value)
        #aggregated_model_path = db.extract_from_table_aggregate(conn,variables['iteration'])

        #if variables['iteration']>5:
         #  db.flush_model_folders(variables['iteration'])      
        
        variables['iteration'] += 1
            
        time.sleep(15)
    
    message_from_master = {'All Iterations Succesfully Completed for':org_ids}

    return message_from_master


def RPC_deepnode(dataframe, token, iteration):

    #get node id
    client_node = do.temp_fix_client()
    org_id = do.find_my_organization_id(client_node)
    node_id = do.find_my_node_id(client_node)

    prm.set_params()

    try:
        if iteration == 0:
            averaged_model_path = os.path.join('/app','initial_weight.h5')
            trained_model_path, model_metrics = run.run_deep_algo(averaged_model_path,org_id,iteration)
        else:
            averaged_model_path = dh.get_model_path(token, iteration-1)
            trained_model_path, model_metrics = run.run_deep_algo(averaged_model_path,org_id,iteration)

        params = {'nodeType':'Node',
        'iteration':iteration,
        'org_id':org_id,
        'training_loss':'/'.join(map(str, model_metrics['training_loss'])),
        'training_dice':'/'.join(map(str, model_metrics['training_dice'])),
        'validation_loss':'/'.join(map(str,model_metrics['validation_loss'])),
        'validation_dice':'/'.join(map(str, model_metrics['validation_dice']))
    }


        if os.path.isfile(trained_model_path):    
            #send averaged model to master
            response = dh.post_model_to_master(params,trained_model_path,token)
            if response==200:
                message_to_server = {'Org id':org_id,
                         'Iteration Completed':iteration,
                         'flag':1}
            else: 
                message_to_server={'Org id':org_id,
                            'Cannot Complete iteration':iteration,
                            'flag':0}
        else:
            message_to_server={'Org id':org_id,
                            'Cannot Complete iteration , no file found in path':iteration,
                            'flag':0}

    except Exception as e: 
        message_to_server={'Org id':org_id,
                            'Cannot Complete iteration': iteration,
                            "Exception":e,
                            'flag':0}

    return message_to_server 
   
