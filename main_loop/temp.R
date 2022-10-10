args <- commandArgs(trailingOnly=TRUE)

#link_hull_incircle --vanilla base_dir final_results.csv
#in_directory <- paste0(args[1], "/temp_pred_leaf")

# read the bitmask for a single leaf csv
leaf_file = gsub("[\n]","",args[1])
leaf <- read.csv(leaf_file)

print("load success")

library (reshape2)

print("success???")