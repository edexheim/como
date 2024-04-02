import torch
import torch.multiprocessing as mp
import queue


def init_gpu(device):
    for i in range(100):
        init_tensor = torch.zeros((8, 192, 256), device="cpu").to(device)
        del init_tensor


def release_data(data):
    del data


def transfer_data(data, device, dtype):
    data_queue = tuple(
        d.to(device=device, dtype=dtype, copy=False) if torch.is_tensor(d) else d
        for d in data
    )
    return data_queue


# Cannot inherit from mp.Queue
class TupleTensorQueue:
    def __init__(self, device, dtype, maxsize=0):
        super().__init__()
        self.queue = mp.Queue(maxsize=maxsize)
        self.device = device
        self.dtype = dtype

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

    def push(self, data, block=True, timeout=None):
        # t1 = time.time()
        data_queue = transfer_data(data, self.device, self.dtype)
        # t2 = time.time()
        # print("transfer data in push: ", t2-t1, self.device)
        self.queue.put(data_queue, block=block, timeout=timeout)

    # def push_recent(self, data, block=True, timeout=None):
    #   data_queue = transfer_data(data, self.device, self.dtype)
    #   if self.queue.full():
    #     old_data = self.queue.get()
    #     release_data(old_data)
    #     self.queue.put(data_queue, block=block, timeout=timeout)
    #   else:
    #     self.queue.put(data_queue, block=block, timeout=timeout)

    def pop(self, block=True, timeout=None):
        try:
            data = self.queue.get(block=block, timeout=timeout)
            return data
        except queue.Empty:
            return None

    def pop_until_latest(self, block=True, timeout=None):
        message = None
        block_loop = block
        while True:
            try:
                message_latest = self.queue.get(block=block_loop, timeout=timeout)
                if message is not None:
                    release_data(message)
                message = message_latest
                # Already got one message so no more blocking regardless
                block_loop = False
            except queue.Empty:
                break

        return message
