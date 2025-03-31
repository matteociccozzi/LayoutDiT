#!/bin/bash

# PubLayNet dataset URL
DATASET_URL="https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz"

BUCKET_NAME="layoutdit"
GCP_PATH="gs://${BUCKET_NAME}/data/publaynet"

# utility to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}


if ! command_exists wget; then
    echo "Error: wget is not installed. Please install it first."
    exit 1
fi

# should exist on gcp VMs by default
if ! command_exists gsutil; then
    echo "Error: gsutil is not installed. Please install Google Cloud SDK first."
    exit 1
fi


TEMP_DIR="/tmp/publaynet"
mkdir -p "${TEMP_DIR}"
cd "${TEMP_DIR}" || exit 1

echo "Downloading dataset..."
wget --show-progress "${DATASET_URL}" -O publaynet.tar.gz || {
    echo "Error: Download failed"
    exit 1
}


echo "Processing and uploading splits..."
tar -xzf publaynet.tar.gz || {
    echo "Error: Failed to extract tar file"
    exit 1
}

for split in train val test; do
    if [ -d "$split" ]; then
        echo "Uploading $split split..."
        gsutil -m cp -r "$split" "${GCP_PATH}/" || {
            echo "Error: Failed to upload $split split"
            exit 1
        }
    else
        echo "Warning: $split directory not found in archive"
    fi
done

cd - > /dev/null
rm -rf "${TEMP_DIR}"

echo "Data gen completed"