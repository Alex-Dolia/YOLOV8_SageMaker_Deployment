import torch, os, json, io, cv2, time, numpy as np
from ultralytics import YOLO
import boto3
   
def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model_name =  'yolov8n-face.pt'
    model = YOLO(f"model/{model_name}")
    return model

def input_fn(request_body, request_content_type): 
    def  write_read_s3_movie(request_body):
        
         s3_bucket, s3_key, image_file_name, local_dir, = request_body.split("|")
         s3_key = s3_key + image_file_name
        
         # Ensure the local directory exists
         if not os.path.exists(local_dir):
                os.makedirs(local_dir)
        
         # Example of writing to the local file system
         local_file_path = os.path.join(local_dir, image_file_name)  
        
         # Download the file from S3
         s3 = boto3.client('s3')

         try:    
                s3.download_file(s3_bucket, s3_key, local_file_path)
         except Exception as e:
                return local_file_path, "Error Loading Image from s3: " + str(e)
        
         is_path_exist = str(os.path.exists(local_dir))
         output = f"!!! The directory {local_dir}) does exist: {is_path_exist}"
                             
         is_file_exist = str(os.path.exists(local_file_path))   
         ss = f"| The file {local_file_path} does exist: {is_file_exist}"
         output += ss                                                    
         return local_file_path, output
    
    local_file_path, status = write_read_s3_movie(request_body)
    assert request_content_type=='text/csv'
    
    orig_image = cv2.imread(local_file_path)
    resized_image = orig_image.copy()
    
    image_height, image_width, _ = resized_image.shape
    model_height, model_width = 300, 300
    x_ratio = image_width/model_width
    y_ratio = image_height/model_height
    resized_image = cv2.resize(resized_image, (model_height, model_width))
    return {"image": resized_image, "status": status, "orig_image": orig_image}
    
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