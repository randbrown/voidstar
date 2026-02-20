#!/bin/bash

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 video1.mp4 video2.mp4"
    exit 1
fi

# Assign arguments to variables for clarity
VIDEO1="$1"
VIDEO2="$2"

# Execute ffplay with hstack filter
ffplay -f lavfi "movie='$VIDEO1'[v0]; movie='$VIDEO2'[v1]; [v0][v1]hstack"
