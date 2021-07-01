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
        self.window_collector = {}
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
                           dnn=wdnn.WDNN(),
                           output_queue=self.output_queue_lvl0,
                           output_barrier=self.output_barrier))

    def send_data_to_windows(self, data: dict):
        for window in self.windows:
            window.input_queue.put(copy.deepcopy(data))
        data.clear()

    def collect_results_from_windows(self):
        for _ in range(len(self.FLAGS.windows)):
            tmp = self.output_queue_lvl0.get()
            self.window_collector.update(tmp)
        self.input_queue_lvl1.put(copy.deepcopy(self.window_collector))
        self.window_collector.clear()
        self.output_barrier.reset()

    def update_window_progresses(self):
        for window in self.windows:
            window.management_queue.put(1)
