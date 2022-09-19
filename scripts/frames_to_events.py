import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from bimvee.exportIitYarp import encodeEvents24Bit
import argparse


def frame2events(input_path, output_path):
    numBins = 10
    step = 0.5 / (numBins)
    dirList = os.listdir(input_path)
    prev_ts = None
    events = []
    ev_img_list = sorted([x for x in dirList if x.__contains__('ec') and os.path.splitext(x)[-1] == '.png'])

    for file in tqdm(ev_img_list):
        with open(os.path.join(input_path, file.split('.')[0] + '.json'), 'r') as jsonFile:
            metadata = json.load(jsonFile)
        timestamp = metadata['timestamp']
        if prev_ts is None:
            prev_ts = timestamp
            continue
        image = plt.imread(os.path.join(input_path, file))
        vCount = np.round(image / step - numBins).astype(int)
        vIndices = vCount.nonzero()
        for y, x in zip(vIndices[0], vIndices[1]):
            num_events = vCount[y, x]
            for v in range(abs(num_events)):
                polarity = 1 if num_events > 0 else 0
                ts_noise = (np.random.rand() - 0.5) / 250
                ts = prev_ts + v * ((timestamp - prev_ts) / abs(num_events)) + ts_noise
                events.append([x, y, ts, polarity])
        prev_ts = timestamp
    events = np.array(events)
    events = events[events[:, 2].argsort()]
    prev_ts = events[0, 2]

    # dataDict = {'x': events[:, 0], 'y': events[:, 1], 'ts': events[:, 2], 'pol': events[:, 3].astype(bool)}
    encodedData = np.array(encodeEvents24Bit(events[:, 2] - events[0, 2], events[:, 0], events[:, 1], events[:, 3].astype(bool))).astype(np.uint32)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'binaryevents.log'), 'wb') as f:
        encodedData.tofile(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract binary events from sequence of frames')
    parser.add_argument('--input', '-i', dest='input_path', type=str, required=True,
                        help='Path to input file')
    parser.add_argument('--output', '-o', dest='output_path', type=str, required=True,
                        help='Path to output file')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # input_path = '/data/Sphere_old/1/photorealistic1'
    # output_path = '/data/Sphere_old/1/'

    frame2events(input_path, output_path)
