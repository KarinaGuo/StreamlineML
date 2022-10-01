args = commandArgs(trailingOnly=TRUE)

#link_hull_incircle --vanilla $base_dir

base_dir = args[1]
in_directory <- paste0(base_dir, "/temp_pred_leaf")

#stats = c("test", "wtf")

#output_file = paste0(base_dir, "/final_results.csv")
#write.table(stats, file = output_file, sep = ”,“, col.names = TRUE,fileEncoding = "UTF-8")

#append = TRUE, 

filenames = "filename"
circle_area_results = "circle_area_results"
mask_area_results = "mask_area_results"
curvature_results = "curvature_results"

stats=(as.data.frame(c(filenames, circle_area_results, curvature_results, mask_area_results), col.names = c(c('file_name','index', 'circle_area_results', 'curvature_results', "mask_area_results")), stringsAsFactors=FALSE))
print(stats)

#write.table(stats, file = output_file, append = TRUE, sep = ”,“, col.names = TRUE)