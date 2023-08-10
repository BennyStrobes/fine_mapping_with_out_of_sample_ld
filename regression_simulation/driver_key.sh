



############################################
# Output data
############################################
# Output root directory
output_root="/n/scratch3/users/b/bes710/fine_mapping_with_out_of_sample_ld/regression_simulation/"

# Directory to save temporary results to
temporary_results_dir=${output_root}"temp_results/"

# Directory to save results to
results_dir=${output_root}"results/"

# Directory to save visualizations to
viz_dir=${output_root}"visualization/"


###########################################
# Run simulation
###########################################
# simulation parameters
n_samples="10000"
n_snps="30"


sh run_simulation.sh $temporary_results_dir $results_dir $n_samples $n_snps