import cv2
import numpy as np
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from flask import request,Flask, make_response
import numpy as np
import joblib
from flask_cors import CORS
from math import ceil


# Define colors for each class for defects
class_colors = {
    0: (102, 204, 102),   # Soft green (for scratches)
    1: (204, 102, 102),   # Muted red (for holes)
    2: (102, 153, 204)    # Calm blue (for discoloration)
}
# Define colors for each class for segmentation
segment_color = {
    0: (255, 165, 0),    # Orange (body) — bright but not harsh, very distinct from greens/blues/reds
    1: (128, 0, 128),    # Purple (wheels) — stands apart from other colors, deep but easy on eyes
    2: (0, 128, 128)     # Teal (backbody) — distinct cyan-greenish, different from orange and purple
}
# Define class names
class_names ={
    0: "Scratches",
    1: "Holes",
    2: "Discoloration"
}
#Open yolo model
yolo_model = YOLO("segmentation_model.pt")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
#Open Mask R-CNN config
cfg = get_cfg()
cfg.merge_from_file("config_defect_detection.yaml")
cfg.MODEL.WEIGHTS = "defect_detection_model.pth"
#Change the confidence threshold into 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
#Open Mask R-CNN predictor
predictor = DefaultPredictor(cfg)

#Open logistic model
logistic_model = joblib.load("root_cause_analysis_model.pkl")
#Open the Multi Label Binarizers
mlbs = joblib.load("multi_label_binarizer.pkl")
mlb_body = mlbs["body"]
mlb_wheels = mlbs["wheels"]
mlb_backbody = mlbs["backbody"]
#Logistic model output dictionary
cause_dict = {
    0: "Assembly transfer arm malfunction (minor collision)",
    1: "Welding machine pressure control failure",
    2: "Damaged cutting tool head",
    3: "Faulty torque limiter in wheel fixture unit",
    4: "Misaligned body panel due to positioning robot error",
    5: "Paint nozzle malfunction in spray system",
    6: "Corrosion from defective metal sheet batch",
    7: "Surface contamination due to inadequate surface prep or paint filtration failure",
    8: "Multiple failures in paint and welding systems",
    9: "Conveyor belt sensor failure causing defect positioning errors"
}



#Process the image
def process_image(image_path):
    #Open the image
    img = cv2.imread(image_path)
    #Get width, height
    width, height = img.shape[:2]
    #Calculate line thickness
    thickness = ceil(max(width, height) * (2/800))
    #Calculate font size
    font_scale = ceil(max(width, height) * (0.5/800))
    #Segmentation with YOLO
    result = yolo_model(image_path)
    #Get the detected classes
    yolo_labels = list(result[0].names.values())
    #Get the detected masks
    yolo_masks = result[0].masks.data.cpu().numpy()
    #Get the detected boxes
    outputs = predictor(img)
    #Draw YOLO results on top of Mask R-CNN results
    for i in range(len(yolo_masks)):
        #Get current mask
        mask = yolo_masks[i]
        
        #Convert the mask to a binary mask
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        #Resize the binary mask to match the image size (height, width)
        binary_mask_resized = cv2.resize(binary_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        #Get contours from the binary mask (for outlining)
        contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #Get the class index for the current object
        class_idx = result[0].boxes.cls[i].item()#Get the class index
        
        #Get color based on the class
        color = segment_color.get(class_idx, (255, 255, 255))  # Default to white if class is not found
        
        #Draw the contours on the image
        cv2.drawContours(img, contours, -1, color, thickness)
        #Get the bounding rectangle for the contour
        x,y,w,h = cv2.boundingRect(contours[0])
        #Get the class name from the class index
        class_name = yolo_labels[int(class_idx)]
        #Draw the class name on the image
        cv2.putText(img, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    #Initialize the defects array
    defects =[[],[],[]]
    #Iterate through the Mask R-CNN instances
    for i in range(len(outputs["instances"])):
        #Get the current instance
        instance = outputs["instances"][i]
        #Get the class index for the Mask R-CNN instance
        class_idx = instance.pred_classes.item()
        #Use the corresponding color for the class
        color = class_colors.get(class_idx, (255, 255, 255)) #White if class is not found
        #Get the class name from the class index
        class_name = class_names[class_idx]
        #Get the mask for the current instance
        mask = instance.pred_masks[0].cpu().numpy()
        #Convert mask to a binary mask
        binary_mask = (mask > 0).astype(np.uint8) * 255
        #Resize the binary mask to match the image size
        binary_mask_resized = cv2.resize(binary_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        #Draw the contours for this instance (class)
        contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Get the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contours[0])
        #Draw the contours on the image with the appropriate color
        cv2.drawContours(img, contours, -1, color, thickness)  # Using class color for each instance
        # Draw the class name above the contour
        cv2.putText(img, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        #Iterate through the Yolo masks
        for mask_idx, mask in enumerate(yolo_masks):
            #Convert the mask to a binary mask
            mask_binary = (mask > 0).astype(np.uint8) * 255
            #Resize the binary mask to match the image size
            mask_binary_resized = cv2.resize(mask_binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            #Get the class index for the current Yolo mask
            class_idx_yolo = int(result[0].boxes.cls[mask_idx].item()  )
            #If both masks overlap
            if cv2.bitwise_and(binary_mask_resized, mask_binary_resized).any():
                #Add the class index to the corresponding defects array if not already present
                defects[class_idx_yolo].append(class_idx) if (class_idx+1) not in defects[class_idx_yolo] else None
                #Print the class name and location
                print(f"{class_name} found on mask {yolo_labels[class_idx_yolo]} at location ({x}, {y})")
                break
        else:
            #Print the class name and location
            print(f"{class_name} found at location ({x}, {y}) but not on any mask")
    #Save the image
    _, encoded_image = cv2.imencode('.jpg', img)
    #Sort the defects
    for i in defects: i.sort()
    #If no defects found, return no cause
    if all(len(d) == 0 for d in defects): return encoded_image, {}
    #Seperate the defects
    defects_on_body, defects_on_wheels, defects_on_backbody = defects
    #Encode the defects
    body_encoded = mlb_body.transform([defects_on_body])
    wheels_encoded = mlb_wheels.transform([defects_on_wheels])
    backbody_encoded = mlb_backbody.transform([defects_on_backbody])
    #Predict the cause
    X_input = np.hstack([body_encoded, wheels_encoded, backbody_encoded])
    probabilities = logistic_model.predict_proba(X_input)[0]
    #Get top 4 indices
    top_indices = np.argsort(probabilities)[::-1][:4]
    #Get top 4 confidences
    top_confidences = probabilities[top_indices]
    top_results = {}
    #Iterate through the top indices
    for i,j in zip(top_indices, top_confidences):
        #Add the cause to the top results dict
        top_results[cause_dict[i]] = float(round(j*100,2))
    #Log the top results
    print(top_results)
    #Return the image
    return encoded_image, top_results

import json

#Create the route /predict
@app.route('/predict', methods = ["POST"])
#Define the predict function
def predict():
    #Get the image
    image_file = request.files['image']
    #Save the image
    image_path = "inputs/"+image_file.filename
    image_file.save(image_path)
    #Process the image
    encoded_image, top_results = process_image(image_path)
    # Create the response
    response = make_response(encoded_image.tobytes())
    # Set the response headers
    response.headers['Content-Type'] = 'image/jpeg'
    response.headers['X-Data'] = json.dumps(top_results)
    response.headers['Access-Control-Expose-Headers'] = 'X-Data'
    # Return the response
    return response


if __name__ == '__main__':
    #Run the app
    app.run(debug=True)