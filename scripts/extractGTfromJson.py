import numpy as np
import matplotlib.pyplot as plt
import os, json
from tqdm import tqdm

dir = '/data/dragon/dragon_translation_z_1_m_s/photorealistic1/'

json_list = sorted([json_files for json_files in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), dir)) if json_files.endswith('.json')])

open(os.path.join(dir, 'ground_truth.csv'), 'w').close()

for file in tqdm(json_list):
    with open(os.path.join(dir, file)) as jsonFile:
        metadata = json.load(jsonFile)
        jsonFile.close()
        position = metadata['objects'][0]['location']
        orientation = metadata['objects'][0]['quaternion_xyzw']
        time = metadata['timestamp']

    with open(os.path.join(dir, 'ground_truth.csv'), 'a') as outFile:
        position = str(position)[1:-1]
        orientation = str(orientation)[1:-1]
        outFile.write(str(time) + ', ' + position + ', ' + orientation +'\n')
        outFile.close()
