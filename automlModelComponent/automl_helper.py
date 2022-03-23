import os
import json
import azureml.automl.core

import joblib

from azureml.core import Workspace, Experiment, Run
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory

def get_workspace():
    run = Run.get_context()
    if (isinstance(run, azureml.core.run._OfflineRun)):
        ws = Workspace.from_config()
    else:
        ws = run.experiment.workspace
    print(f"Retrieved access to workspace {ws}")
    return ws

def get_automl_run(workspace, experiment, run_id):
    try:
        experiment = Experiment(workspace, experiment)
        automl_run = Run(experiment, run_id)

        if ('runTemplate' not in automl_run.properties or automl_run.properties['runTemplate'] != "automl_child"):
            raise RuntimeError(f"Run with run_id={run_id} is a not an AutoML run!")

        # Get parent run
        parent_run = automl_run.parent
        while (parent_run.parent is not None):
            parent_run = parent_run.parent
        
        if (parent_run.type != 'automl'):
            raise RuntimeError(f"Only AutoML runs are supported, this run is of type {parent_run.type}!")
    except Exception as e:
        raise

    return automl_run

def load_automl_model(automl_run):
    print("Downloading AutoML model...")
    automl_run.download_file('outputs/model.pkl', output_file_path='./')
    model_path = './model.pkl'
    model = joblib.load(model_path)
    return model

def write_prediction_dataframe(dir_path, dataframe):
    print("Writing predictions back...")
    os.makedirs(dir_path, exist_ok=True)
    save_data_frame_to_directory(dir_path, dataframe)