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


DATA_DIR="./publaynet"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}" || exit 1

echo "Downloading dataset..."
wget --show-progress "${DATASET_URL}" -O publaynet.tar.gz || {
    echo "Error: Download failed"
    exit 1
}


cd publaynet/

tar -xzf publaynet.tar.gz || {
    echo "Error: Failed to extract tar file"
    exit 1
}

# after extracting the data will be in publaynet/ dir with this structure:
#train/	Images in the training subset
#val/	Images in the validation subset
#test/	Images in the testing subset
#train.json	Annotations for training images
#val.json	Annotations for validation images
#LICENSE.txt	Plaintext version of the CDLA-Permissive license
#README.txt	Text file with the file names and description

echo "Current working directory: $(pwd)"
echo "Contents of the directory (ls -a):"
ls -a

# upload all contents to gcp bucket
gsutil -m cp -r . gs://layoutdit/data/publaynet/

echo "Data gen completed"