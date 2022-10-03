import os, sys
sys.path.append("/home/botml/code/dev/")
import linking_functions
import math
import shutil

base_dir = os.getcwd()
print ("Base directory set as: ", base_dir)

os.makedirs(train_d2_data) #Move labelled training data into here ##later link this to the mainloop, change dir name
os.makedirs(trim_d2_data)
os.makedirs(trim_d2_validation)
os.makedirs(model)
os.makedirs(model/d2)

train_d2_data_dir = os.path.join(base_dir, 'train_d2_data/')
trim_d2_data_dir = os.path.join(base_dir, 'trim_d2_data/')
trim_d2_validation_dir = os.path.join(base_dir, 'trim_d2_validation/')
input_d2_test_dir = os.path.join(base_dir, 'input_d2_test')

#Cut to bounding box from input_d2_data to trim_d2_data
linking_functions.cut_focal_box (train_d2_data_dir, trim_d2_data_dir, 'bb', ['Leaf90', 'Leaf100']) ######### need to fix this!!
json_file = glob.glob(os.path.join(input_d2_data_dir, '*.json'))) #replace dir with image dir 
for imfiles in json_file:
  shutil.move(imfiles, trim_d2_data_dir)

#Move 20% of files to validation
file_list = sorted(glob.glob(os.path.join(trim_d2_data, '*'))) #replace dir with image dir 
num = len(file_list))*0.2
move_num = math.ceil((num/2)*2)

for idx, files in enumerate(file_list):
  if idx <= move_num:
    in_file = os.path.join(trim_d2_data_dir, files)
    shutil.move(in_file, trim_d2_validation_dir)

#Making lm2coco.py of the training and validation files  
linking_functions.lm2coco(trim_d2_data_dir, 'trim_d2_data.json', 'poly', ['Leaf90', 'Leaf100']) ######### need to fix this!!
linking_functions.lm2coco(trim_d2_validation_dir, 'trim_d2_validation.json', 'poly', ['Leaf90', 'Leaf100']) ######### need to fix this!!

#Train detectron model
linking_functions.train_leaf(base_dir)

#Preparing test data
os.makedirs(input_d2_test)
linking_functions.lm2coco(input_d2_test_dir, 'd2_test.json', 'poly', ['Leaf90', 'Leaf100']) ######### need to fix this!!
os.makedirs(d2_pred)

#Predicting
os.makedirs(pred_leaf)
os.makedirs(pred_leaf/bitmask)
linking_functions.predict_leaf (base_dir, "model")
linking_functions.extract_leaves (base_dir, "model", "Y") #Manually classify images to Y/N

#Training classifier model
os.makedirs(input_classifier_data) #Move classified training data into here
os.makedirs(input_classifier_data/N)
os.makedirs(input_classifier_data/Y)
os.makedirs(model/classifier)

linking_functions.train_classifier (base_dir)

#Testing classifier model
class_res = open('classifier_results_test.csv', 'x')
class_res.close
linking_functions.predict_from_classifier(base_dir, "model") ##fix output
### Code to use bitmask as validation mask


