# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn
import mxnet as mx
import json
from PIL import Image
from io import BytesIO
import numpy as np
import io
import json
import logging
from operator import itemgetter
from typing import Optional
import boto3


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    print("extract model information here! ")
    net = gluon.SymbolBlock.imports(
        "%s/model-symbol.json" % model_dir,
        ["data"],
        "%s/model-0000.params" % model_dir,
    )
    return net




logger = logging.getLogger("infcustom")
logger.info("Loading custom inference handlers")
s3client = boto3.client("s3")



class S3ObjectSpec:
    """Utility class for parsing an S3 location spec from a JSON-able dict"""

    def __init__(self, spec: dict):
        if "URI" in spec:
            print("URI")
            if not spec["URI"].lower().startswith("s3://"):
                raise ValueError("URI must be a valid 's3://...' URI if provided")
            bucket, _, key = spec["URI"][len("s3://") :].partition("/")
        else:
            print("bucket and key ",spec)
            bucket = spec["bucket"]
            print("bucket",bucket)
            key = spec["key"]
            print("key",key)
        if not (bucket and key and isinstance(bucket, str) and isinstance(key, str)):
            raise ValueError(
                "Must provide an object with either 'URI' or 'Bucket' and 'Key' properties. "
                f"Parsed bucket={bucket}, key={key}"
            )
        self.bucket = bucket
        self.key = key
        


def read_image_from_s3(bucket, key, region_name='ap-southeast-1'):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3 = boto3.resource('s3', region_name=region_name)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    return np.array(im)[np.newaxis, np.newaxis,  :]
        

def input_fn(input_bytes, content_type: str):
    """Deserialize and pre-process model request JSON

    Requests must be of type application/json. See module-level docstring for API details.
    """
    logger.info(f"Received request of type:{content_type}")
    if(content_type != "application/json"):
        raise ValueError("Content type must be application/json")

    req_json = json.loads(input_bytes)

    s3_input = req_json["S3Input"]
    
    if(s3_input):
        try:
            
            s3_input = S3ObjectSpec(s3_input)
            
        except ValueError as e:
            raise ValueError(
                "Invalid Request.S3Input: If provided, must be an object with 'URI' or 'Bucket' "
                "and 'Key'"
            ) from e
        logger.info(f"Fetching S3Input from s3://{s3_input.bucket}/{s3_input.key}")
        s3_image_key=f"s3://{s3_input.bucket}/{s3_input.key}"
        img_np = read_image_from_s3(s3_input.bucket, s3_input.key)
       
        
        doc_json = json.dumps(img_np.tolist())
       
    else:
        try:
            "Content" in req_json
        except ValueError as e:
            raise ValueError(
                "Invalid Input: If provided, must be an object with json"
            ) from e
        
        if("Content" in req_json):
            doc_json = req_json["Content"]
            req_root_is_doc = False
        else:
            doc_json = req_json
            req_root_is_doc = True

    s3_output = req_json["S3Output"]
    if(s3_output):
        print("s3_output")
        try:
            s3_output = S3ObjectSpec(s3_output)
        except ValueError as e:
            raise ValueError(
                "Invalid Request.S3Output: If provided, must be an object with 'URI' or 'Bucket' "
                "and 'Key'"
            ) from e
        

    

    return {
        "doc_json": doc_json,
     
        "s3_output": s3_output,
      
    }


def output_fn(prediction_output, accept):
    """Finalize model result JSON.

    Requests must accept content type application/json. See module-level docstring for API details.
    """
    if(accept != "application/json"):
        raise ValueError("Accept header must be 'application/json'")

    doc_json, s3_output = itemgetter("doc_json", "s3_output")(prediction_output)

    if(s3_output):
        logger.info(f"Uploading S3Output to s3://{s3_output.bucket}/{s3_output.key}")
        s3client.upload_fileobj(
            io.BytesIO(json.dumps(doc_json).encode("utf-8")),
            Bucket=s3_output.bucket,
            Key=s3_output.key,
        )
        return json.dumps(
            {
                "Bucket": s3_output.bucket,
                "Key": s3_output.key,
                "URI": f"s3://{s3_output.bucket}/{s3_output.key}",
            }
        ).encode("utf-8")
    else:
        return json.dumps(doc_json).encode("utf-8")

def predict_fn(input_data: dict, model: dict):
    """Classify WORD blocks on a Textract result using a LayoutLMForTokenClassification model

    Parameters
    ----------
    input_data : { doc_json,  s3_output }
        Parsed JSON of Image data, plus additional control parameters.
    model : { config, device, model, tokenizer }
        The core token classification model, tokenizer, config (not used) and PyTorch device.

    Returns
    -------
    doc_json : Union[List, Dict]
        Predicted value of tissue classification 
        s3_output : S3ObjectSpec for output
        Passed through from input_data
    """
    doc_json,  s3_output = itemgetter(
        "doc_json",  "s3_output"
    )(input_data)
    print("doc_json:")
    print("doc_json:",s3_output)
    ## conduct prediction here
    
    
    data=json.loads(doc_json)
    #data=np.array(data)
    nda = nd.array(data)
    
    #print(nda)
    output = model(nda)
    prediction = nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    #print("response_body: ",response_body)
    return {"doc_json": response_body, "s3_output": s3_output}

