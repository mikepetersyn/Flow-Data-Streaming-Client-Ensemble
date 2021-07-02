'''
ensemble client (multiple time windows)
'''
import copy
import gzip
import json
import os
import pickle
from datetime import datetime
import dnn_manager
import wdnn
import meta_dnn
from log import init_logger
from utils import create_tf_dataset_lvl0, create_tf_dataset_lvl1
from utils import jsonStrings2int

if __name__ == '__main__':  # only init logger for main process
  init_logger(__file__)

from window_manager import WindowManager
import client
import defaultParser
import csv
from processor import Flow_Processor
import tensorflow as tf
import math
import time
import numpy as np
from constants import *
import sys

if __name__ == '__main__':
  print('TensorFlow version {}'.format(tf.__version__))
  if tf.__version__ != '1.14.0': print('tested with tensorflow version 1.14', file=sys.stderr)

import_path = '/media/data/tnsm_preprocessed_blocks'

export_path = '/media/data/evaluation_data'
if not os.path.isdir(export_path):
    os.makedirs(export_path)



# TODO: logging is broken

class StreamClientEnsemble(client.Stream_Client):
  ''' training class (inherited class from Stream_Client)
    1. build base estimators (fully-connected dnn) and ensemble strategy
    2. start thread to receive data (from inherited class)
    3. treat each block as a global variable and fork thread again into multiple childs that represent a windowed process
  '''

  def __init__(self):
    client.Stream_Client.__init__(self)

    #-------------------------------------------------------------------------------------------------------- ARG PARSER
    self.big_block_count = 40
    self.block_count = 0
    self.FLAGS = None
    self.arg_parse()
    #-------------------------------------------------------------------------------------------------- OUTPUT PARAMETER
    self.output_filename = None
    self.output_file = None
    self.csv_writer = None
    self.output_filename = self.FLAGS.output_file
    if self.FLAGS.output_file:  # create CSV writer
      self.output_file = open(self.output_filename, 'w')
      self.csv_writer = csv.writer(self.output_file)

    self.result = {}

    self.pool = multiprocessing.Pool(NUM_PROCESSES)  # process pool for preprocessing of a data block
    self.window_manager = WindowManager()

    self.meta_dnn = meta_dnn.MetaDNN()
    # self.connect()
    # self.start()  # start the connection which use the overwritten process method to process the received data
    self._loop()  # start training and testing loop

  def arg_parse(self):
    ''' create command line parser '''
    parser = defaultParser.create_default_parser()
    self.FLAGS, _ = parser.parse_known_args()
    defaultParser.printFlags(self.FLAGS)

  @staticmethod
  def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

  @staticmethod
  def get_model_acc(prob_arr, labels):
    accuracy = {classifier : np.mean(np.equal(
            np.argmax(prob_arr[classifier], axis=1),
            np.argmax(labels, axis=1)).astype(np.float32), axis=0)
                     for classifier in prob_arr.keys()}
    return accuracy

  @staticmethod
  def load_gzip(filename):
      with gzip.open(filename, 'rb') as file:
          data = pickle.load(file)
      return data

  def load_from_buffer(self):
      dataset = self.result.get('dataset')
      class_weights = self.result.get('class_weighting_train')
      self.result.clear()
      return dataset, class_weights

  def get_mini_block_size(self, block):
      return int(block.shape[0]*round(self.FLAGS.buffer_block_size/block.shape[0], 2))

  @staticmethod
  def ts_to_daytime_feature(timestamp):
      dt = datetime.fromtimestamp(timestamp)
      return [np.float32(dt.hour/24), np.float32(dt.minute/60)]

  @staticmethod
  def create_lvl1_input_vector(timestamps, prob_arr, labels):
      # daytimes = [self.ts_to_daytime_feature(ts) for ts in timestamps]
      prob_arr = np.concatenate(([prob_arr[window] for window in sorted(prob_arr.keys())]), axis=1)
      return np.concatenate((prob_arr, labels), axis=1)

  def _loop(self):
    """
    * check after each batch if new data is available
    * fork subprocess that invokes 'process_block' of each windowDNN
    * wait for results of all windows to join and apply ensemble method
    """
    while True:  # load block, training and testing loop
      # if len(self.result) == 0:
      #   time.sleep(1)
      #   continue

      file_name=f'{import_path}/block_{self.big_block_count}.gz'
      self.result = self.load_gzip(file_name)

      dataset, class_weights = self.load_from_buffer()

      mini_block_size = self.get_mini_block_size(dataset)

      num_batches = math.ceil(dataset.shape[0] / mini_block_size)
      print(f'big block is split into {num_batches} mini blocks')

      for chunk in self.chunker(dataset, mini_block_size):
          timestamps = chunk[:, 0]
          labels = chunk[:, -3:]

          mini_block = {
              'dataset': chunk,
              'class_weights': class_weights
          }

          self.window_manager.send_data_to_windows(mini_block)

          while True:
              if self.window_manager.input_queue_lvl1.empty():
                  if self.meta_dnn.dataset is None:
                      time.sleep(1)
                  elif not self.FLAGS.fixed_num_epochs:
                      self.meta_dnn.train()
                  else:
                      time.sleep(1)
              else:
                  accs, ys, y_pred_probs = self.window_manager.input_queue_lvl1.get()

                  labels_out = np.array(ys[0]).reshape(-1, 3)[:chunk.shape[0]]

                  lvl1_input = self.create_lvl1_input_vector(
                      timestamps,
                      y_pred_probs,
                      labels_out)

                  self.meta_dnn.dataset = create_tf_dataset_lvl1(lvl1_input)
                  self.meta_dnn.class_weighting_block = class_weights
                  accs_meta, y_pred_probs_meta = self.meta_dnn.test()

                  y_pred_probs.update({99: y_pred_probs_meta})
                  accs.update({99: accs_meta})

                  self.meta_dnn.train()

                  print('test accuracy at superblock {} - block {} ({}): {}'.format(
                      self.big_block_count,
                      self.block_count,
                      str(datetime.fromtimestamp(timestamps[0])),
                      [(window, accs[window]) for window in sorted(accs.keys())]))

                  if self.csv_writer:
                      row = [timestamps[0], self.big_block_count, self.block_count]
                      for window in sorted(accs.keys()):
                          row.append(accs[window])
                      self.csv_writer.writerow(row)
                      self.output_file.flush()

                  eval_data = {
                      'accuracies': accs,
                      'y_pred_prob': y_pred_probs,
                      'timestamps': timestamps,
                      'labels': labels_out}

                  file = f'{export_path}/superblock-{self.big_block_count}_block-{self.block_count}.pkl'
                  with open(file, 'wb') as f:
                      pickle.dump(eval_data, f, pickle.HIGHEST_PROTOCOL)
                  break
          self.block_count += 1
      self.big_block_count += 1
      self.window_manager.update_window_progresses()
      continue

    if self.csv_writer:
        self.output_file.close()


  def process(self, data):
    ''' start the preprocessing step for the next incoming block (called by base class)

    @param data: input data from stream server (np.array)
    @return: None (never stop the processing)
    '''
    Flow_Processor(data, self.result, self.FLAGS, self.pool).start()


if __name__ == '__main__':
  client = StreamClientEnsemble()
  client.join()
