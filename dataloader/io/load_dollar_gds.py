# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import glob
import numpy as np
import os
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from dataloader.sample import Sample


# ----------------------------------------------------------------------------------------------------------------------
def download_dollar_gds(root):
    """
    Downloads the $1-GDS dataset.
    Dataset homepage: http://depts.washington.edu/acelab/proj/dollar/index.html

    :param root: where to download the data
    """
    dataset_url = 'http://depts.washington.edu/acelab/proj/dollar/xml.zip'

    target_dir = os.path.join(root)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    print("Downloading from '{}'...".format(dataset_url))
    filename = dataset_url.split('/')[-1]
    filepath = os.path.join(root, filename)
    urllib.request.urlretrieve(dataset_url, filepath)

    print("\tExtracting '{}'...".format(filepath))
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(root)


# ----------------------------------------------------------------------------------------------------------------------
def parse_xml(file):
    """
    Reads one sample file

    :param file: the file to read
    :return: a Sample instance
    """
    gesture = ET.parse(file).getroot()
    label = ''.join([i for i in gesture.attrib['Name'] if not i.isdigit()])
    pts = []

    for pt in gesture:
        pts.append([
            float(pt.attrib['X']),
            float(pt.attrib['Y']),
        ])

    return Sample(x=np.asarray(pts, dtype=np.float32),
                  label=label,
                  unique_id=file)


# ----------------------------------------------------------------------------------------------------------------------
def load_dollar_gds(root):
    """
    Loads the $1-GDS dataset.

    :param root: the root path of the data
    :return: a list of Sample objects
    """
    dataset_dir = os.path.join(root, 'xml_logs')

    # Does the dataset exist? If not, download it
    if not os.path.exists(dataset_dir):
        print('$1-GDS dataset not found!')
        download_dollar_gds(root)

    dirs = glob.glob(os.path.join(dataset_dir, '*'))
    samples = []

    for dr in dirs:
        # Don't process the pilot session's data
        if 'pilot' in dr:
            continue

        for speed in ['medium']:  # Can optionally add 'slow' and 'fast' to this list too

            curr_files = glob.glob(os.path.join(dr, speed, '*.xml'))

            for file in curr_files:
                samples += [parse_xml(file)]

    return samples
