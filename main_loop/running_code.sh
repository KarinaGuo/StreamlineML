#!/bin/bash
#mkdir model/
#mkdir model/classifier 
#echo "Making directories: /model, /model/d2, /model/classifier"

touch classifier_results_test.csv final_results.csv
echo "Making file: /classifier_results_test.csv, /final_results.csv"

base_dir=`pwd`
echo "Base directory set as" $base_dir

for i in  $(cat file_list.txt); do 
	touch temp_filter.csv 
	touch temp_classifier_results_test.csv
	echo "Making files: /classifier_results_test.csv, /temp_filter.csv"
	
	mkdir temp_image_subset 
	echo "Making directories: /temp_image_subset"

	mkdir temp_pred 
	mkdir temp_pred_leaf/subf 
	echo "Making directories: /temp_pred, /temp_pred_leaf"
	
	echo copying ${i}; 
	ln -s ${i} temp_image_subset/ 
	python predict_leaf.py $base_dir "main"
	python extract_leaves.py $base_dir "main"
  python removing_files.py 
  python running_R.py $base_dir "final_results.csv"
	python predict_from_classifier.py $base_dir "main"
	rm -r temp*
done