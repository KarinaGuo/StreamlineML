#This code is for profiling!

#!/bin/bash
#mkdir model/
#mkdir model/classifier 
#echo "Making directories: /model, /model/d2, /model/classifier"

conda activate pytorch
touch log.txt
echo "Made log.file" >> log.txt

base_dir=`pwd`
echo "Base directory set as" $base_dir

curr_date=`date`
echo $curr_date >> log.txt

touch classifier_results_test.csv 
echo "Making file: /classifier_results_test.csv"
#python /home/botml/code/dev/main_loop/adding_header.py $base_dir

mkdir temp_image_subset image_subset_lists
echo "Making folder: temp_image_subset, image_subset_lists"

mkdir classifier_training_data

python /home/botml/code/dev/main_loop/tmp/batching_files_temp.py $base_dir 3
batch_list=`ls -d $PWD/image_subset_lists/*`

echo "batches are" ${batch_list}

for batch in $batch_list; do 
	echo "starting" ${batch} >> log.txt
	now_time=`date`
	echo $now_time >> log.txt
	conda activate pytorch
	touch temp_filter.csv 
	touch temp_classifier_results_test.csv
	echo "Making files: /classifier_results_test.csv, /temp_filter.csv" 

  mkdir temp_image_subset 
	echo "Making directories: /temp_image_subset"

	mkdir temp_pred 
	mkdir temp_pred_leaf
	mkdir temp_pred_leaf/subf 
	echo "Making directories: /temp_pred, /temp_pred_leaf"
	
	echo "" >> log.txt
  for i in $(cat ${batch}); do echo copying ${i} >> log.txt; image=`basename ${i}`; ln -s ${i} temp_image_subset/$image; done 
	echo "images copied"
  
  echo "" >> log.txt
  echo ${batch} "system time for predict_leaf" >> log.txt
  (time python /home/botml/code/dev/main_loop/predict_leaf.py $base_dir "main") 2>> log.txt
	echo "leaves predicted"
	
  echo "" >> log.txt
  echo ${batch} "system time for extract_leaves" >> log.txt
  (time python /home/botml/code/dev/main_loop/extract_leaves_mt.py $base_dir "main" "Y" >> log.txt) 2>> log.txt
	echo "leaves cropped"
	
  echo "" >> log.txt
  echo ${batch} "system time for predict_from_classifier" >> log.txt
  (time python /home/botml/code/dev/main_loop/predict_from_classifier.py $base_dir "main") 2>> log.txt 
	echo "classifier classed"  
 
	python /home/botml/code/dev/main_loop/removing_files.py $base_dir
	
  #echo "" >> log.txt
  #echo ${batch} "system time to activate MLpredictions" >> log.txt
  conda activate MLpredictions
  
  echo "" >> log.txt
  echo ${batch} "system time for running_R and internal R processes" >> log.txt
	(time python /home/botml/code/dev/profiling/code/running_R.py $base_dir >> log.txt) 2>> log.txt
  echo "leaves traited" 

	rm -r temp*
	echo ${batch} "removed temporary files"
	echo ${batch} "completed!" >> log.txt
	echo "batch completed :)"
  echo "" >> log.txt
done