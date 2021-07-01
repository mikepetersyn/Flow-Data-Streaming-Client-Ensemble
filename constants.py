'''
constants definition
'''
import multiprocessing

LOG_LEVEL = 'DEBUG'

SERVER_ADDRESS = '192.168.178.33'

SERVER_PORT = 11338
SOCKET_TIMEOUT = 60 * 10  # seconds
MAX_SOCKET_TIMEOUTS = 3
BUFFER_SIZE = 4096
NUM_PROCESSES = 1 #multiprocessing.cpu_count() - 1  # or a fix number of processes
