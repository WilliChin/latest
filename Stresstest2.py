


import json
from pathlib import Path
import os
import logging
import pandas as pd

# %% Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])

logging.info("Starting logging")


# %% Paths
path_input = Path(os.environ.get("INPUTS", "/data/inputs"))
path_output = Path(os.environ.get("OUTPUTS", "/data/outputs"))
path_logs = Path(os.environ.get("LOGS", "/data/logs"))
dids = json.loads(os.environ.get("DIDS", '[]'))
assert dids, f'no DIDS are defined, cannot continue with the algorithm'

did = dids[0]
input_files_path = Path(os.path.join(path_input, did))
input_files = list(input_files_path.iterdir())
first_input = input_files.pop()
# assert len(input_files) == 1, "Currently, only 1 input file is supported."
path_input_file = first_input
logging.debug(f'got input file: {path_input_file}, {did}, {input_files}')
path_output_file = path_output / 'prime.csv'

# %% Check all paths
assert path_input_file.exists(), "Can't find required mounted path: {}".format(path_input_file)
assert path_input_file.is_file() | path_input_file.is_symlink(), "{} must be a file.".format(path_input_file)
assert path_output.exists(), "Can't find required mounted path: {}".format(path_output)
# assert path_logs.exists(), "Can't find required mounted path: {}".format(path_output)
logging.debug(f"Selected input file: {path_input_file} {os.path.getsize(path_input_file)/1000/1000} MB")
logging.debug(f"Target output folder: {path_output}")


# %% Load data
logging.debug("Loading {}".format(path_input_file))

with open(path_input_file, 'rb') as fh:
    df = pd.read_csv(fh)

logging.debug("Loaded {} records into DataFrame".format(len(df)))


arr = df["num"].to_numpy()

primes = []
nprimes = []

for num in arr:
    if num > 1:
        for k in range(2, int(num ** 0.5) + 1):
            if num % k == 0:
                nprimes.append(num)
                break
        else:
            primes.append(num)
    else:
        print(num, " can't be checked, because its smaller than 1")
        
        
logging.debug("Built summary of records.")

primes.to_csv(path_output_file)

logging.debug("Wrote results to {}".format(path_output_file))

logging.debug("FINISHED ALGORITHM EXECUTION")



