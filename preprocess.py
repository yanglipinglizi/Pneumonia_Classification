from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
from pathlib import Path
import cv2

# We need the csv file containnig the labels
labels = pd.read_csv("Data/stage_2_train_labels.csv")

# Remove duplicate entries
labels = labels.drop_duplicates("patientId")

# Saving path to dicom file and also the path were we want to store our processed npy files
ROOT_PATH = Path("Data/stage_2_train_images")
SAVE_PATH = Path("Data/check_test")

# Taking a look at few images
counter = 0
fig, axis = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    for j in range(3):
        patient_id = labels.patientId.iloc[counter]
        dcm_path = ROOT_PATH/patient_id
        dcm_path = dcm_path.with_suffix(".dcm")
        dcm = pydicom.read_file(dcm_path).pixel_array
        
        label = labels["Target"].iloc[counter]
        
        axis[i][j].imshow(dcm, cmap="bone")
        axis[i][j].set_title(label)
        counter += 1
plt.show()

''' In order to efficiently handle our data in the Dataloader, we convert the X-Ray images stored in the DICOM format to numpy arrays. Afterwards we compute the overall mean and standard deviation of the pixels of the whole dataset, for the purpose of normalization.
Then the created numpy images are stored in two separate folders according to their binary label:
* 0: All X-Rays which do not show signs of pneumonia
* 1: All X-Rays which show signs of pneumonia
'''

sums = 0
sums_squared = 0

for k, patient_id in enumerate(tqdm(labels.patientId)):
    dcm_path = ROOT_PATH/patient_id  # Create the path to the dcm file
    dcm_path = dcm_path.with_suffix(".dcm")  # And add the .dcm suffix
    
    # Read the dicom file with pydicom and standardize the array
    dcm = pydicom.read_file(dcm_path).pixel_array / 255  
        
    # Resize the image as 1024x1024 is way to large to be handeled by Deep Learning models at the moment so reshaping to 224x224
    # In order to use less space when storing the image converting it to float16
    dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
    
    # Retrieve the corresponding label
    label = labels.Target.iloc[k]
    
    # 4/5 train split, 1/5 val split
    train_or_val = "train" if k < 24000 else "val" 
        
    current_save_path = SAVE_PATH/train_or_val/str(label) # Define save path and create if necessary
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path/patient_id, dcm_array)  # Save the array in the corresponding directory
    
    normalizer = dcm_array.shape[0] * dcm_array.shape[1]  # Normalize sum of image
    if train_or_val == "train":  # Only use train data to compute dataset statistics
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (np.power(dcm_array, 2).sum()) / normalizer

# Calculating mean and std
mean = sums / 24000
std = np.sqrt(sums_squared / 24000 - (mean**2))

print(f'Mean: {mean}, Standard Deviation: {std}')