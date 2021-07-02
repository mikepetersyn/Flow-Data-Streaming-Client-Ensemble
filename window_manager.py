import copy
import multiprocessing
from typing import Optional, List
import defaultParser
import wdnn
from utils import *
from dnn_manager import DNNManager


class WindowManager:

    def __init__(self, ):
        self.FLAGS = None
        self.arg_parse()
        self.windows: List[DNNManager] = []
        self.y_pred_prob_collector = {}
        self.ys_collector = {}
        self.acc_collector = {}
        self.output_queue_lvl0: multiprocessing.Queue = multiprocessing.Queue()
        self.input_queue_lvl1: multiprocessing.Queue = multiprocessing.Queue()
        self.output_barrier = multiprocessing.Barrier(
            parties=len(self.FLAGS.windows),
            action=self.collect_results_from_windows)
        self.init_windows()

    def arg_parse(self):
        """ create command line parser """
        parser = defaultParser.create_default_parser()
        self.FLAGS, _ = parser.parse_known_args()
        defaultParser.printFlags(self.FLAGS)

    def init_windows(self):
        for window_size in self.FLAGS.windows:
            self.windows.append(
                DNNManager(window_size=window_size,
                           output_queue=self.output_queue_lvl0,
                           output_barrier=self.output_barrier))

    def send_data_to_windows(self, data: dict):
        for window in self.windows:
            window.input_queue.put(copy.deepcopy(data))
        data.clear()

    def collect_results_from_windows(self):
        for _ in range(len(self.FLAGS.windows)):
            acc, ys, y_pred_probs = self.output_queue_lvl0.get()
            self.acc_collector.update(acc)
            self.ys_collector.update(ys)
            self.y_pred_prob_collector.update(y_pred_probs)
        self.input_queue_lvl1.put((copy.deepcopy(self.acc_collector), copy.deepcopy(self.ys_collector), copy.deepcopy(self.y_pred_prob_collector)))
        self.acc_collector.clear()
        self.ys_collector.clear()
        self.y_pred_prob_collector.clear()
        self.output_barrier.reset()

    def update_window_progresses(self):
        for window in self.windows:
            window.management_queue.put(1)
