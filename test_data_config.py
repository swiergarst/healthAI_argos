import json
from pathlib import Path

current_path = Path(__file__).parent


with open(f'{current_path}/argos_layout/argos_layout/assets/slices_validation_modified.json', "r") as fp:
    train_slices_config = json.load(fp)


print(train_slices_config.keys())

for key in train_slices_config.keys():
    print(train_slices_config[key][0])
    break

# print(train_slices_config)

# print(current_path)
# # total_paths = 0

### change paths to have the correct path through all user-specific folders ###
# to_replace = "/Users/palan001/Desktop/Health-AI/ARGOS/data/nsclc_radiomics_v4/"

# new_config = {}
# for key in train_slices_config.keys():
#     new_config[key] = []
#     for path in train_slices_config[key]:
#         new_path = path.replace(to_replace, f'{current_path}/argos_layout/')
#         new_config[key].append(new_path)

# with open(f'{current_path}/argos_layout/argos_layout/assets/slices_validation_modified.json', "w") as fp:
#     json.dump(new_config, fp)

