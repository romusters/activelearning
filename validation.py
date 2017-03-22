import pandas as pd
import numpy as np
import os, sys
import features
import config
root, data_path, model_path, vector_path = config.get_paths()
print root

import pandas as pd
import dataset
dset = dataset.Dataset(root)