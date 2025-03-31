import requests
import tarfile
import os
from os import path

fname = 'examples.tar.gz'
url = 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/' + fname
r = requests.get(url)
open(fname , 'wb').write(r.content)

tar = tarfile.open(fname)
tar.extractall()
tar.close()

data_path = "examples/"
path.exists(data_path)

os.remove(fname)

