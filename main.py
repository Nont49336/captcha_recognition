import pandas as pd
from preprocessing_pipeline import *
from char_processing import processed_label
import glob
from model import CharacterCNN,get_num_clases,get_classes
from Levenshtein import distance
import torch
from func import binary2torch


test_path = "D:/CS4243_miniproj/captcha_img/test"
dataset_path = "D:/CS4243_miniproj/train_dataset/dataset2"
csv_save_path = "D:/CS4243_miniproj/test_result/test2.csv"
model_pth = "D:/CS4243_miniproj/models/character_cnn4.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = get_classes(dataset_path)
test_df = pd.DataFrame(columns=["path","prediction"])
debug = True

num_classes = get_num_clases(dataset_path)
print("num_classes:",num_classes)

#model loading
model = CharacterCNN(num_classes)
model.load_state_dict(torch.load(model_pth,weights_only=True,map_location="cuda:0"))
# model = torch.load(model_pth)

if device == "cuda":
    model.to(device)

# Finish Model inference

# retrieve test folder imgs
test_list = glob.glob(f"{test_path}/*.png")

# initialize dataframe to save the result
for i in test_list:
    # print(i)
    label = processed_label(i)
    img_preprocessed = img_preprocessing_cleon(i)
    ord_rois,ord_areas = img_postprocessing(img_preprocessed,char_padding=1)
    prediction = ""
    for roi in ord_rois:
        # cv2.imshow("roi",roi)
        # cv2.waitKey(0)
        roi = binary2torch(roi)

        # print(roi.shape)

        if device == "cuda":
            roi = roi.to(device)
        output = model(roi)
        _,predicted = torch.max(output, 1)
        char = classes[predicted]
        # print(char)
        prediction+=char
    print("saving:",i,"prediction:",prediction,"distance:",distance(prediction,label))
    add_row = {"path":i,"prediction":prediction}
    test_df = pd.concat([test_df, pd.DataFrame([add_row])], ignore_index=True)
test_df.to_csv(csv_save_path,sep=',')
    
