#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import random

import imgviz
import labelme
from os import walk

try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--annotations_dir", help="Datasets Directories", required=True)
    args = parser.parse_args()

    if not osp.exists(args.annotations_dir):
        print("Annotations directory not exists:", args.output_dir)
        sys.exit(1)
    
    ORIGINAL_PATH = args.annotations_dir
    TEST_PATH = ORIGINAL_PATH+"/test"
    if osp.exists(TEST_PATH):
        os.rmdir(TEST_PATH)
    os.makedirs(TEST_PATH)
    TRAIN_PATH = ORIGINAL_PATH+"/train"
    if osp.exists(TRAIN_PATH):
        os.rmdir(TRAIN_PATH)
    os.makedirs(TRAIN_PATH)

    _, _, filenames = next(walk(ORIGINAL_PATH))
    num_to_select = int(len(filenames)*0.1) # Select 10 percent to test
    test_filenames = random.sample(filenames, num_to_select)
    train_filenames = set(filenames) - set(test_filenames)
    for filename in test_filenames:
        os.replace("{}/{}".format(ORIGINAL_PATH, filename), "{}/{}".format(TEST_PATH, filename)) 
    for filename in train_filenames:
        os.replace("{}/{}".format(ORIGINAL_PATH, filename), "{}/{}".format(TRAIN_PATH, filename)) 

if __name__ == "__main__":
    main()
