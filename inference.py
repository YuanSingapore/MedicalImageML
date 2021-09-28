# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn
import json
#import numpy as np

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    print("--- inside model_fn")
    net = gluon.SymbolBlock.imports(
        "%s/model-symbol.json" % model_dir,
        ["data"],
        "%s/model-0000.params" % model_dir,
    )
    return net

import numpy as np
import mxnet as mx

# def transform_fn(net, data, input_content_type='list', output_content_type='json'):
#    """
#   Transform a request using the Gluon model. Called once per request.
 #   :param net: The model.
#    :param data: The request payload.
#    :param input_content_type: The request content type.
#    :param output_content_type: The (desired) response content type.
#    :return: response payload and content type.
#    """
#    # we can use content types to vary input/output handling, but
#    # here we just assume json for both
    
#    print("------- inside transform_fn----------")
#    nda = nd.array(data)
    #print(nda)
#    output = net(nda)
#    prediction = nd.argmax(output, axis=1)
#    response_body = json.dumps(prediction.asnumpy().tolist()[0])
#    return response_body, output_content_type



def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled numpy array"""
    print("------- inside input_fn----------")
    print('request_content_type: ', request_content_type)
    
    print('request_body: ', request_body)
    if request_content_type == 'application/json':
        print("json dumps")
        parsed = json.loads(request_body)
        
        return parsed
    else:
        
        parsed = nd.array(request_body)
        return parsed
       
        
    
def predict_fn(data, net):
    print("------- inside input_fn----------")
    data=json.loads(data)
    #data=np.array(data)
    nda = nd.array(data)
    #print(nda)
    output = net(nda)
    prediction = nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    
    
    print("------- outside of input_fn----------")
    return response_body


def output_fn(prediction_output, accept):
    """Finalize model result JSON.

    Requests must accept content type application/json. See module-level docstring for API details.
    """
    if accept != "application/json":
        raise ValueError("Accept header must be 'application/json'")
    
    
    return prediction_output
   
