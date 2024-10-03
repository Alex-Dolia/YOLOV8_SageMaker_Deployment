import torch, os, json, io, cv2, time, numpy as np
import base64
from ultralytics import YOLO
import boto3
   
def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model_name =  'yolov8n-face.pt'
    model = YOLO(f"model/{model_name}")
    return model

def input_fn(request_body, request_content_type):
    """
    Preprocess the incoming request.

    Parameters:
    request_body (str): The body of the incoming request.
    request_content_type (str): The content type of the incoming request.

    Returns:
    np.array: The preprocessed image.
    """
    # Check the content type
    if request_content_type == 'application/json':
        # Parse the JSON object
        request_data = json.loads(request_body)

        # Get the base64-encoded image
        image_base64 = request_data.get('instances', [{}])[0].get('b64', None)

        if image_base64 is None:
            raise ValueError("The request doesn't contain a base64-encoded image.")

        # Decode the base64 string
        image_data = base64.b64decode(image_base64)

        # Convert the image data to a numpy array
        image_np = np.frombuffer(image_data, dtype=np.uint8)

        # Decode the numpy array as an image
        orig_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        resized_image = orig_image.copy()
    
        image_height, image_width, _ = resized_image.shape
        model_height, model_width = 300, 300
        x_ratio = image_width/model_width
        y_ratio = image_height/model_height
        resized_image = cv2.resize(resized_image, (model_height, model_width))
        
        status = "" # we need to update it using exception but for now it is fine
        return {"image": resized_image, "status": status, "orig_image": orig_image}
    raise ValueError(f"Unsupported content type: {request_content_type}")
    
def predict_fn(input_data, model):
    image = input_data["image"].copy()
    print("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(image)

    return {"prediction": result, "status": input_data["status"], "orig_image": input_data["orig_image"]} 
        
def output_fn(prediction_output, content_type):
    class NumpyEncoder(json.JSONEncoder):
          def default(self, obj):
              if isinstance(obj, np.ndarray):
                 return obj.tolist()
              return json.JSONEncoder.default(self, obj)
    
    infer = {}
    infer["status"] = prediction_output["status"]
    infer["orig_image"] = prediction_output["orig_image"]
    
    prediction_output = prediction_output["prediction"]
    print("Executing output_fn from inference.py ...")
    
    for result in prediction_output:
        if 'boxes' in result._keys and result.boxes is not None:
            if torch.cuda.is_available():
               infer['boxes'] = result.boxes.cpu().numpy().data.tolist() 
            else:    
               infer['boxes'] = result.boxes.numpy().data.tolist() 
    return json.dumps(infer, cls = NumpyEncoder)