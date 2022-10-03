import os, glob
import sys
sys.path.append("/home/botml/code/dev")
import linking_functions

base_dir = os.getcwd()
print ("Base directory set as: ", base_dir)

#Making folders
temp_pred = os.path.join(base_dir, "temp_pred/")
temp_pred_leaf = os.path.join(base_dir, "temp_pred_leaf/subf/")
temp_image_subset = os.path.join(base_dir, "temp_image_subset")

os.makedirs(temp_pred)
os.makedirs(temp_pred_leaf)
print ("Making directories: /temp_pred, /temp_pred_leaf")
os.makedirs(temp_image_subset)
print ("Making directories: /temp_image_subset")

#Making files
temp_filter = open('temp_filter.csv', 'x')
temp_filter.close
temp_classifier_results_test = open('temp_classifier_results_test.csv', 'x')
temp_classifier_results_test.close
print ("Making files: /classifier_results_test.csv, /temp_filter.csv")

file_list = glob.glob('./*.jpg') #replace dir with image dir 
text_file = open("file_list.txt", 'w')
for image in file_list:
  text_file.write(image + "\n")

text_file.close()
  
image_list = os.path.join(base_dir, "file_list.txt")
with open(image_list) as filelist:
    for img in filelist:
      os.symlink(img, temp_image_subset)
      print(f"Copying {img})
	    linking_functions.predict_leaf(base_dir, "main")
	    linking_functions.extract_leaves(base_dir, "main", "Y")
      linking_functions.removing_files() 
      running_R.py (base_dir)
	    linking_functions.predict_from_classifier(base_dir, "main")
      list_remove = glob.glob(os.path.join("base_dir/temp*"), recursive=True)
      for filepath in list_remove:
        os.remove(filepath)

