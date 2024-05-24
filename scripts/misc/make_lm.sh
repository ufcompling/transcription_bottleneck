#!/bin/bash

# Define the root directory containing the 9 subfolders
root_dir="/mmfs1/data/leferran/scripts/Polysynthetic/data"
lmplz_script="/mmfs1/data/leferran/scripts/Polysynthetic/kenlm/build/bin/lmplz"
# Loop over each subdirectory within the root directory
for subfolder in "$root_dir"/*/; do
    # Define the split folder within the current subfolder
    split_dir="${subfolder}split"

    # Check if the split folder exists
    if [ -d "$split_dir" ]; then
        # Loop over the "min", "max", and "rand" directories
        for category in min max rand; do
            # Define the category directory
            category_dir="${split_dir}/${category}"

            # Define the output file name
            output_file="${split_dir}/combined_${category}.txt"

            # Create or clear the output file
            > "$output_file"

            # Loop over each text file in the current category directory and concatenate its content to the output file
            for file in "$category_dir"/train/*.txt; do
                cat "$file" >> "$output_file"
                echo >> "$output_file" # Add a newline to separate files
            done
            echo $split_dir/$(basename ${output_file} .txt).arpa
            $lmplz_script -S 1G -o 3  <$output_file > $split_dir/$(basename ${output_file} .txt).arpa
            echo "Created $output_file from the contents of ${category_dir}"
        done
    else
        echo "No split directory found in ${subfolder}"
    fi
done

