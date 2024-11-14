import easyocr
import pandas as pd
import glob
from Levenshtein import distance
import cv2

file_path = "D:/CS4243_miniproj/captcha_img/test"
# model_path = "C:/Users/nont/Desktop/test_LPR/EasyOCR/trainer/saved_models/th_test/th_pretrained_g1.pth"
# recog_network_yaml_path = "C:/Users/nont/Desktop/test_LPR/test_model/test_model"
lpr_df = pd.DataFrame(columns=["path","prediction"])
reader = easyocr.Reader(lang_list=["en"])
# img = cv2.imread("C:/Users/nont/Desktop/test_LPR/th_train/valid/1671.jpg")
# img = cv2.resize(img,(64,600),(-1,-1),interpolation=cv2.INTER_AREA)
# res = reader.readtext(img,
#                       detail=0,
#                       add_margin=0.5
#                       )
# print(res)

file_list = sorted(glob.glob(file_path+'/*.png'))
# ['2ก"8874', 'กรุงเทหมหาบคร']
# ['4ขส5014']
for i in file_list:
    res = reader.readtext(  i
                            ,detail=0
                            ,allowlist=['0','1','2','3','4'
                                     ,'5','6','7','8','9','A','B'
                                     ,'C','D','E','F','G','H','I','J','K','L','M'
                                     ,'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                                     'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
                            # ,add_margin=0.5
                        )
    if len(res) == 1:
            print("digit:",res[0])
            res = res[0].replace(" ","").lower()
            add_row = {'path':i,'prediction':res}
            lpr_df = pd.concat([lpr_df, pd.DataFrame([add_row])], ignore_index=True)

lpr_df.to_csv("result_easyocr.csv",sep=";",index=False,encoding="utf-16")


