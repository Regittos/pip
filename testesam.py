
#requiriments
#pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
#pip install opencv-python pycocotools matplotlib onnxruntime onnx
#pip install git+https://github.com/facebookresearch/segment-anything.git
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


sam_checkpoint = "sam_vit_h_4b8939.pth"  


sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam.to(device)
sam.eval()


image_path = "chairs.png"  
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


predictor = SamPredictor(sam)


predictor.set_image(image)


input_point = np.array([[500, 375]]) 
input_label = np.array([1])  


masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)


plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    plt.imshow(mask, alpha=0.5)
plt.axis('off')
plt.show()
