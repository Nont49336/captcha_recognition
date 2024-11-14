from preprocessing_pipeline import img_preprocessing,img_postprocessing,img_preprocessing_cleon
from char_processing import char_tokenizer,processed_label
from func import save_img,mat_show,data_representation
import glob
import os

train_folder_pth = "D:/CS4243_miniproj/captcha_img/train"
save_folder_pth = "D:/CS4243_miniproj/train_dataset/dataset2"
img_id = 0
os.makedirs(save_folder_pth,exist_ok=True)

img_list = glob.glob(f"{train_folder_pth}/*.png")
for i in img_list:
    img = img_preprocessing_cleon(i)
    # mat_show(img)
    ord_rois,ord_areas = img_postprocessing(img,char_padding=1)
    imgs,labels = char_tokenizer(ord_rois,label=i)
    if imgs == None:
        # save_img(img,processed_label(i))
        continue
    else:
        for img,label in zip(imgs,labels):
            save_dir = f"{save_folder_pth}/{label}/{img_id}.png"
            print(save_dir)
            # print(save_dir)
            img_id += 1
            save_img(img,save_dir)
data_representation(save_folder_pth)
