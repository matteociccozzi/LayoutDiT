# LayoutDiT
Layout parsing using a vision transformer backbone

## Development setup
This project uses uv to manage python dependencies, python virtual environments, and python executable themselves. 
You only need this tool, nothing else!

Install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/) then run `make sync`.
If you add a package run `uv add xyz`, otherwise if you modify the pyproject.toml file directly make sure to run 
`make lock` and then `make sync`

## PubLayNet
The dataset being used for fine-tuning is available [here](https://github.com/ibm-aur-nlp/PubLayNet?tab=readme-ov-file).

## GCP 
### Bootstrap dataset
Create a small instance first, we will use this to copy over the publay dataset ot a GCP bucket
```
gcloud compute instances create data-vm \
  --zone=us-west1-a \
  --machine-type=e2-standard-2 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=200GB
```

then ssh into it: 
```gcloud compute ssh data-vm --zone=us-west1-a```

Clone the repo, and run `data_gen.sh`. This will download the dataset from the CDN setup and then untar and upload to 
GCP bucket so that we can use it for training.

### Deep Learning VM, GPU Quotas etc

### Running training entrypoint
#### Locally
Setup application auth login: `gcloud auth application-default login`, this is needed to allow the application to load 
data from the gcloud bucket. 
It is not feasible to run training on your local machine, this is more of a sanity check should you want to run training
from your local machine to make sure the code runs through fine. 

#### GCP VM
Create a Deep Learning VM, with your desired specifications, then you can connect via ssh as: 
```
gcloud compute ssh --project layoutdit --zone us-west2-b deeplearning-1-vm -- -L 8080:localhost:8080
```

To setup the remote debugger first update your ssh config running `gcloud compute config-ssh --project=layoutdit`
Then create an SSH Interpreter on port 22, the config should be picked up automatically. 

We want to avoid reading data from gs directly during training so we should use 
`gcsfuse layoutdit /input/data/layoutdit` 

This will give us lower per-worker overhead, built-in caching, and full POSIX compatibility so we avoid the timeouts
and complexity of per-worker GCS clients.


# TODO
- Add support for distributed training
- Refactor trainer to put a custom contex manager for profiler
- tests
- yaml config for improved human readabilty? 