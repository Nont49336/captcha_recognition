import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms,datasets
def mat_show(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(12, 6))
    plt.imshow(rgb_image)
    plt.axis('off')  # Turn off axis
    plt.show()

def data_representation(train_folder):
    # Path to the main directory containing folders
    main_directory = train_folder

    # Dictionary to hold folder names and file counts
    folder_file_count = {}

    # Loop through each folder in the main directory
    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Count the number of files in the folder
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            # Add to dictionary
            folder_file_count[folder_name] = file_count

    # Prepare data for plotting
    folder_file_count = dict(sorted(folder_file_count.items()))
    # folder_file_count = sorted(folder_file_count)
    folders = list(folder_file_count.keys())
    file_counts = list(folder_file_count.values())

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(folders, file_counts, color='skyblue')
    plt.xlabel('Folders')
    plt.ylabel('Number of Files')
    plt.title(os.path.basename(train_folder))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{os.path.dirname(train_folder)}/{train_folder}_represent.png', format='png', dpi=300)
    plt.show()

def save_img(img,path):
    print(os.path.dirname(path))
    os.makedirs(os.path.dirname(path),exist_ok=True)
    try:
        cv2.imwrite(path,img)
    except:
        with open("img_empty.txt",'+a') as f:
            f.writelines(f"{path}\n")
            f.close()
        pass
    print("saving:",path)
    # os.makedirs(path,exist_ok=True)
    # cv2.imwrite(path)
def binary2torch(img):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Resize images to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])
    img = img * 255
    img_pil = Image.fromarray(img)
    img_transformed = transform(img_pil).unsqueeze(0) 
    return img_transformed

if __name__ == "__main__":
    data_representation("D:/CS4243_miniproj/train_dataset/dataset2")