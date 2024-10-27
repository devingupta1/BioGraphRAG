import os
import pandas as pd
import json

BASE_PATH = "../biokg_nebula"
files = os.listdir(BASE_PATH)

entity_to_entity_type_map = {}

for file in files:
    if file in ["biokg.links.csv", "entity_to_entity_type.json"]:
        continue
    entity_type = file.split(".")[2] # file naming convention
    df = pd.read_csv(BASE_PATH + f"/{file}", header=None)
    e1_list = df[0].tolist()
    e2_list = df[2].tolist()
    for e1 in e1_list:
        entity_to_entity_type_map[e1] = entity_type
    for e2 in e2_list:
        entity_to_entity_type_map[e2] = entity_type

with open(BASE_PATH + "/entity_to_entity_type.json", "w") as f:
    json.dump(entity_to_entity_type_map, f)
