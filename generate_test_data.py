import numpy as np
import pandas as pd
from pathlib import Path
import json


# get path of current directory
current_path = Path(__file__).parent

# read in config file
with open(f'{current_path}/config.json', "r") as fp:
    config = json.load(fp)

img_width = config['patch_shape'][0]
img_height =  config['patch_shape'][1]
n_slices =  config['patch_shape'][2]
n_images = 6
n_classes = config['num_classes']
seed = 7 # i've used seed 7 for mock train data, seed 42 for mock val data

# just use a normal distribution generator from numpy
rng = np.random.default_rng(seed=seed)
images = rng.normal(size = (img_width, img_height, n_slices, n_images))
labels = rng.integers(0, high = n_classes, size = n_images)


# pandas expects a dataframe, so we have to flatten these images into a dataframe
images_flattened = images.reshape((n_images, -1))
col_names = ([f"img_pix_{i}" for i in range(img_width * img_height * n_slices)])
img_df = pd.DataFrame(data = images_flattened, columns=col_names)
img_df['labels'] = labels

print('saving to csv')
img_df.to_csv("mock_data.csv", index = False)

# TODO: generate labels as well