import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import skimage
from skimage.segmentation import find_boundaries
from PIL import Image
import time

device ='cuda' if torch.cuda.is_available() else 'cpu'
sam = sam_model_registry ["vit_h"] (checkpoint ="./sam_vit_h_4b8939.pth")
sam.to( device = device )

mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.96)

def SAMAug(tI , mask_generator):
    masks = mask_generator.generate(tI)
    if len(masks) == 0:
        return
    tI= skimage.img_as_float (tI)

    BoundaryPrior = np.zeros (( tI. shape [0] , tI. shape [1]))
    BoundaryPrior_output = np.zeros ((tI.shape [0] , tI. shape [1]))
    
    Objects_first_few =  np.zeros (( tI. shape [0] , tI. shape [1]))
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    idx=1
    for ann in sorted_anns:        
        if ann['area'] < 50:
            continue
        if idx==51:
            break
        m = ann['segmentation']
        color_mask = idx
        print(color_mask)
        Objects_first_few[m] = color_mask
        idx=idx+1

    for maskindex in range(len(masks)):
        thismask =masks[ maskindex ][ 'segmentation']
        mask_=np.zeros (( thismask.shape ))
        mask_[np.where( thismask == True)]=1
        BoundaryPrior = BoundaryPrior + find_boundaries (mask_ ,mode='thick')

    BoundaryPrior [np.where( BoundaryPrior >0) ]=1
    BoundaryPrior_index=np.where(BoundaryPrior >0)
    Objects_first_few[BoundaryPrior_index]= 0  
    BoundaryPrior_output [np.where( BoundaryPrior >0) ]=255
    BoundaryPrior_output = BoundaryPrior_output.astype(np.uint8) 
    return BoundaryPrior_output,Objects_first_few  

# directory_name='./loveDA/Urban/images_png/'
directory_name='./ISPRS_dataset/Vaihingen/top/'

# img_list=[f for f in os.listdir(directory_name)]
# img_list=sorted(img_list)
# start_time=time.time()
# for img_input in img_list:
#     if img_input.endswith('.png'):
#         img_name=img_input.split(".")[0]
#         image_type=".png"
#         image = Image.open(directory_name+img_input)
#         image = np.array(image)
#         print(type(image))
#         print(image.shape)
#         BoundaryPrior_output, Objects_first_few=SAMAug(image, mask_generator)
#         image_boundary = Image.fromarray(BoundaryPrior_output)
#         Objects_first_few = Objects_first_few.astype(np.uint8)
#         image_objects = Image.fromarray(Objects_first_few)
#         image_boundary.save("./SAM/loveDA_obj_data/"+img_name+'_Boundary'+image_type)
#         image_objects.save("./SAM/loveDA_obj_data/"+img_name+'_objects'+image_type)

img_list=[f for f in os.listdir(directory_name)]
img_list=sorted(img_list)
start_time=time.time()
for img_input in img_list:
    if img_input.endswith('.tif'):
        img_name=img_input.split(".")[0]
        image_type=".tif"
        image = Image.open(directory_name+img_input)
        image = np.array(image)
        print(type(image))
        print(image.shape)
        new_file_boundary = "./ISPRS_dataset/Vaihingen/sam_boundary_merge/"+img_name+'_Boundary'+image_type
        new_file_object = "./ISPRS_dataset/Vaihingen/V_merge/"+img_name+'_objects'+image_type
        # os.makedirs(new_directory_boundary, exist_ok=True)
        # os.makedirs(new_directory_object, exist_ok=True)
        BoundaryPrior_output, Objects_first_few=SAMAug(image, mask_generator)
        image_boundary = Image.fromarray(BoundaryPrior_output)
        Objects_first_few = Objects_first_few.astype(np.uint8)
        image_objects = Image.fromarray(Objects_first_few)
        image_boundary.save(new_file_boundary)
        image_objects.save(new_file_object)
end_time = time.time()
run_time = end_time - start_time
print(f"Runing time: {run_time} second.")