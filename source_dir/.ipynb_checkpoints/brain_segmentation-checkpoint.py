from __future__ import absolute_import
import boto3
import base64
import json
import io
import os
import argparse
import mxnet as mx
from mxnet import nd
import numpy as np
mx.test_utils.download("https://raw.github.com/drj11/pypng/main/code/png.py", "png.py")
import png
#from sagemaker_mxnet_training.training_utils import save

import tarfile
from iterator import DataLoaderIter
from losses_and_metrics import avg_dice_coef_metric
from models import build_unet, build_enet
import logging

import brain_segmentation_s3_transform

logging.getLogger().setLevel(logging.INFO)

#from inference import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--class-weights", type=list, default=[1.35, 17.18,  8.29, 12.42])
    parser.add_argument("--network", type=str, default="local")
    parser.add_argument("--instance_type", type=str, default="ml.p3.2xlarge")
    parser.add_argument("--instance_count", type=int, default=1)

    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--training-env", type=list, default=json.loads(os.environ["SM_TRAINING_ENV"]))
    print(parser.parse_args())
    print('parser.parse_args().instance_type: ',parser.parse_args().instance_type)
    print("parser.instance_count: ", parser.parse_args().instance_count)
    print("parser.network: ", parser.parse_args().network)
    return parser.parse_args()

    
###############################
###     Training Loop       ###
###############################
    
def train(current_host, training_channel, testing_channel, hyperparameters, hosts, model_dir,num_cpus, num_gpus):
    logging.info("=========== Inside Train ===========")
    logging.info(mx.__version__)
    
    # Set context for compute based on instance environment
    if num_gpus > 0:
        ctx = [mx.gpu(i) for i in range(num_gpus)]
    else:
        ctx = mx.cpu()

    # Set location of key-value store based on training config.
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    
    # Get hyperparameters
    batch_size = hyperparameters.get('batch_size', 16)        
    learning_rate = hyperparameters.get('lr', 1E-3)
    beta1 = hyperparameters.get('beta1', 0.9)
    beta2 = hyperparameters.get('beta2', 0.99)
    epochs = hyperparameters.get('epochs', 100)
    num_workers = hyperparameters.get('num_workers', 6)
    num_classes = hyperparameters.get('num_classes', 4)
    class_weights = hyperparameters.get(
        'class_weights', [[1.35, 17.18, 8.29, 12.42]])
    class_weights = np.array(class_weights)
    network = hyperparameters.get('network', 'unet')
    assert network == 'unet' or network == 'enet', '"network" hyperparameter must be one of ["unet", "enet"]'
    
    # Locate compressed training/validation data
    train_dir = training_channel
    print(train_dir)
    validation_dir = testing_channel
    print(validation_dir)
    train_tars = os.listdir(train_dir)
    validation_tars = os.listdir(validation_dir)
    # Extract compressed image / mask pairs locally
    for train_tar in train_tars:
        with tarfile.open(os.path.join(train_dir, train_tar), 'r:gz') as f:
            f.extractall(train_dir)
    print("extracted all the files for training!")
    for validation_tar in validation_tars:
        with tarfile.open(os.path.join(validation_dir, validation_tar), 'r:gz') as f:
            f.extractall(validation_dir)
    print("extracted all the files for validation!")
    # Define custom iterators on extracted data locations.
    train_iter = DataLoaderIter(
        train_dir,
        num_classes,
        batch_size,
        True,
        num_workers)
    print("DataLoaderIter done for training!")
    validation_iter = DataLoaderIter(
        validation_dir,
        num_classes,
        batch_size,
        False,
        num_workers)
    
    # Build network symbolic graph
    if network == 'unet':
        print("unet, class_weights =", class_weights)
        sym = build_unet(num_classes, class_weights=class_weights)
    else:
        print("enet")
        sym = build_enet(inp_dims=train_iter.provide_data[0][1][1:], num_classes=num_classes, class_weights=class_weights)
    logging.info("Sym loaded")
    
    # Load graph into Module
    net = mx.mod.Module(sym, context=ctx, data_names=('data',), label_names=('label',))
    
    # Initialize Custom Metric
    dice_metric = mx.metric.CustomMetric(feval=avg_dice_coef_metric, allow_extra_outputs=True)
    logging.info("Starting model fit")
    
    # Start training the model
    net.fit(
        train_data=train_iter,
        eval_data=validation_iter,
        eval_metric=dice_metric,
        initializer=mx.initializer.Xavier(magnitude=6),
        optimizer='adam',
        optimizer_params={
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2},
        num_epoch=epochs)
    
    # Save Parameters
    net.save_params('params')
    
    # Build inference-only graphs, set parameters from training models
    if network == 'unet':
        sym = build_unet(num_classes, inference=True)
    else:
        sym = build_enet(
            inp_dims=train_iter.provide_data[0][1][1:], num_classes=num_classes, inference=True)
    net = mx.mod.Module(
        sym, context=ctx, data_names=(
            'data',), label_names=None)
    
    # Re-binding model for a batch-size of one
    net.bind(data_shapes=[('data', (1,) + train_iter.provide_data[0][1][1:])])
    net.load_params('params')
    logging.info("Save model {}/model".format(model_dir))
    
    save(model_dir, net,hosts=hosts)
    #save(model_dir, net)
    
    return net

SYMBOL_PATH = 'model-symbol.json'
PARAMS_PATH = 'model-0000.params'
SHAPES_PATH = 'model-shapes.json'


def save(model_dir, model, current_host=None, hosts=None):
    """Save an MXNet Module to a given location if the current host is the scheduler host.
    This generates three files in the model directory:
    - model-symbol.json: The serialized module symbolic graph.
        Formed by invoking ``module.symbole.save``.
    - model-0000.params: The serialized module parameters.
        Formed by invoking ``module.save_params``.
    - model-shapes.json: The serialized module input data shapes in the form of a JSON list of
        JSON data-shape objects. Each data-shape object contains a string name and
        a list of integer dimensions.
    Args:
        model_dir (str): the directory for saving the model
        model (mxnet.mod.Module): the module to be saved
    """
    current_host = current_host or os.environ['SM_CURRENT_HOST']
    hosts = hosts or json.loads(os.environ['SM_HOSTS'])

    if current_host == scheduler_host(hosts):
        model.symbol.save(os.path.join(model_dir, SYMBOL_PATH))
        model.save_params(os.path.join(model_dir, PARAMS_PATH))

        signature = [{'name': data_desc.name, 'shape': [dim for dim in data_desc.shape]}
                     for data_desc in model.data_shapes]
        with open(os.path.join(model_dir, SHAPES_PATH), 'w') as f:
            json.dump(signature, f)

def scheduler_host(hosts):
    """Return which host in a list of hosts serves as the scheduler for a parameter server setup.
    Args:
        hosts (list[str]): a list of hosts
    Returns:
        str: the name of the scheduler host
    """
    return hosts[0]
    
    


def test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    return metric.get()



        
        
        
if __name__ == "__main__":
    args = parse_args()
    import argparse
    print(args.hosts)
    #print('instance_type=',args['instance_type'])
    #print('instance_count=', args['instance_count'])
    #parser.add_argument('--instance_count', type=int, default=os.environ['SM_NUM_GPUS'])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    if(args.instance_type=='local'):
        print("local testing")
        num_gpus = 0 #int(os.environ["SM_NUM_GPUS"])
    else:
        num_gpus=1
        num_cpus=0
    
    hyperparameters = args.training_env["hyperparameters"]
    logging.info("batch-size = {}".format(hyperparameters["batch-size"]))
    print("hyperparameters:",hyperparameters)
    print("num_gpus ==",num_gpus,num_cpus )
    print("args.train==", args.train)
    print("args.test==", args.test)


    print(num_gpus)
    train(
        args.current_host,
        args.train,
        args.test,
        hyperparameters,
        args.hosts,
        args.model_dir,
        num_cpus,
        num_gpus
    )
    