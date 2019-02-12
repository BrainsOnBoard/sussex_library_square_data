#!/bin/bash

# Lists of memory types and variants to test
MEMORY_TYPES=( PerfectMemory PerfectMemoryConstrained )
VARIANTS=( mask unwrapped skymask )

# Create directory to hold benchmark results
mkdir -p benchmark_results
rm -f benchmark_results/output.csv

echo "Route name, memory type, variant, RMSE" >> benchmark_results/output.csv

# Loop through directories containing routes
for r in routes/*; do
    # Read decimate variable from shell script in directory
    source $r/decimate.sh

    # Get name of route
    ROUTE_NAME=$(basename $r)
    
    # Loop through memory types and variants and calculate vector field
    for m in "${MEMORY_TYPES[@]}"; do
        for v in "${VARIANTS[@]}"; do
            echo "${ROUTE_NAME}, ${m}, ${v}"
            
            # Run vector field renderer, tailing and cutting out RMSE value from end
            RMSE=`./vector_field --route=$ROUTE_NAME --variant=$v --decimate-distance=$DECIMATE_DISTANCE --output-image=benchmark_results/grid_image_${ROUTE_NAME}_${m}_${v}.png --output-csv=benchmark_results/output_${ROUTE_NAME}_${m}_${v}.csv --memory-type=$m | tail --lines=1 | cut -d ':' -f 2`
            
            # Write CSV line to output
            echo "${ROUTE_NAME}, ${m}, ${v}, ${RMSE}" >> benchmark_results/output.csv
        done
    done
done
