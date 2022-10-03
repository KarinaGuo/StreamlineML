#!/bin/bash
#training the machine
mkdir input_d2_data #Move labelled training data into here ##later link this to the mainloop, change dir name
mkdir trim_d2_data
mkdir trim_d2_validation
mkdir model 
mkdir model/d2

tmp_1=`ls temp_image_subset/ | wc -l`
tmp_2=`echo "2* $tmp_1 * 0.2" | bc`
move_number=`printf '%.*f\n' 0 $tmp_2`

base_dir=`pwd`

###Detectron model
#Prepare images for training
python cut_focal_box.py base_dir --focalbox 'bb' --classes 'Leaf100' --classes 'Leaf90'
cp -r input_d2_data/*.json trim_d2_data/
ls -q temp_image_subset/ | head -$move_number | xargs -i mv temp_image_subset/{} trim_d2_validation/
python lm2coco.py trim_d2_data --output trim_d2_data.json --classes 'Leaf100' --classes 'Leaf90' --polyORbb 'bb'
python lm2coco.py trim_d2_validation --output trim_d2_validation.json --classes 'Leaf100' --classes 'Leaf90' --polyORbb 'bb'

#Train detectron model
python train_leaf.py base_dir

#Preparing test data
mkdir input_d2_test
python lm2coco.py input_d2_test --output d2_test.json --classes 'Leaf100' --classes 'Leaf90' --polyORbb 'bb'
mkdir d2_pred

#Predicting
mkdir pred_leaf
mkdir pred_leaf/bitmask
python predict_leaf.py base_dir model
python extract_leaves.py base_dir model Y#Manually classify images to Y/N

#Training classifier model
mkdir input_classifier_data #Move classified training data into here
mkdir input_classifier_data/N
mkdir input_classifier_data/Y
mkdir model/classifier
python train_classifier base_dir

#Testing classifier model
touch classifier_results_test.csv
python predict_from_classifier base_dir model ##fix output
### Code to use bitmask as validation mask