# -*- coding: utf-8 -*-

import os
from scipy.misc import imread
import numpy as np

TRAIN_INPUTS = ""
TRAIN_TARGETS = ""

class Dataset(object):

    def __init__(self):
        self.ids = self.load_ids()

    def load_ids(self):
        ids = os.listdir(TRAIN_INPUTS)
        ids = [id[:-4] for id in ids]
        return ids

    def load_images(self, l):
        output = []
        for i in l:
            im = np.expand_dims(imread(os.path.join(TRAIN_INPUTS, i+".jpg")), axis=0)
            im = im.swapaxes(1, 3)
            output.append(im)
        output = np.vstack(output)
        return output

    def load_masks(self, l):
        output = []
        for i in l:
            im = np.expand_dims(imread(os.path.join(TRAIN_TARGETS, i+"_mask.gif")), axis=0)
            im = im.swapaxes(1, 3)
            output.append(im)
        output = np.vstack(output)
        return output

    def iter_batch(self, batch_size=32):
        for i in xrange(0, len(self.ids), batch_size):
            batch_ids = self.ids[i:i+batch_size]
            images = self.load_images(batch_ids)
            masks = self.load_masks(batch_ids)
            yield images, masks