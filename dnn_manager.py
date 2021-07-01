import copy
import multiprocessing
import queue
import time
from typing import List
import defaultParser
import wdnn
from utils import create_tf_dataset_lvl0


class DNNManager(multiprocessing.Process):
    def __init__(self, window_size, dnn: wdnn.WDNN, output_queue, output_barrier):
        multiprocessing.Process.__init__(self)

        self.FLAGS = None
        self.arg_parse()

        self.window_size = window_size

        self.dnn = dnn

        self.management_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.input_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.output_queue: multiprocessing.Queue = output_queue
        self.output_barrier: multiprocessing.Barrier = output_barrier

        self.batch_results: List = []

        self.block_num = 0
        self.online = True

        self.start()

    def arg_parse(self):
        """ create command line parser """
        parser = defaultParser.create_default_parser()
        self.FLAGS, _ = parser.parse_known_args()
        defaultParser.printFlags(self.FLAGS)

    def check_window_progress(self):
        try:
            signal = self.management_queue.get(block=False)
        except queue.Empty:
            return
        if signal is not None:
            self.block_num += 1
        if self.block_num % self.window_size == 0:
            self.online = not self.online
        print(self.window_size, ': ',self.online, self.block_num)


    def run(self):
        self.dnn.build_dnn()
        while True:
            if self.window_size != 0:
                self.check_window_progress()
            if self.input_queue.empty():
                if self.dnn.dataset is None:
                    time.sleep(1)
                else:
                    self.online_training()
            else:
                self.load_from_queue()
                self.batch_results.append(self.dnn.test())
                self.output_queue.put({self.window_size: copy.deepcopy(self.batch_results.pop())})
                self.output_barrier.wait()
                self.online_training()

    def online_training(self):
        if self.online:
            self.dnn.train()
        else:
            time.sleep(1)

    def load_from_queue(self):
        mini_block = self.input_queue.get()
        self.dnn.dataset = create_tf_dataset_lvl0(mini_block.get('dataset'))
        self.dnn.class_weighting_block = mini_block.get('class_weights')
