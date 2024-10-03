import torch, os, json, io, cv2, time
from ultralytics import YOLO
import boto3
   
def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model_name =  'yolov8n-face.pt'
    model = YOLO(f"model/{model_name}")
    return model

def input_fn(request_body, request_content_type): 
    #
    assert request_content_type=='text/csv'
    image_file_name = request_body
    
    orig_image = cv2.imread(image_file_name)
    image_height, image_width, _ = orig_image.shape
    model_height, model_width = 300, 300
    x_ratio = image_width/model_width
    y_ratio = image_height/model_height
    resized_image = cv2.resize(orig_image, (model_height, model_width))
    return {"image": resized_image}
    
def predict_fn(input_data, model):
    image = input_data["image"].copy()
    print("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(image)

    return {"prediction": result} 
        
def output_fn(prediction_output, content_type):
    prediction_output = prediction_output["prediction"]
    print("Executing output_fn from inference.py ...")
    infer = {}
    for result in prediction_output:
        if 'boxes' in result._keys and result.boxes is not None:
            if torch.cuda.is_available():
               infer['boxes'] = result.boxes.cpu().numpy().data.tolist() 
            else:    
               infer['boxes'] = result.boxes.numpy().data.tolist() 
    return json.dumps(infer)