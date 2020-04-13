import numpy as np


class Kernel(object):
    def __init__(self):
        self.memory = None
        self.d, self.k = None, None  # d: distance (num neighbour layers), k: num. cells
        self.remove_idxs = None  # position indices to remove some of the cells from kernel.
        self.conclusion_idx = None  # the cell position index being investigated with its neighbours

    def _get_kernel(self, frame, row, col):
        return frame[row-self.d:row+self.d+1, col-self.d:col+self.d+1]

    def _memory_handler(self, row, col, neighs):
        if self.memory:
            memo = self.memory.learn(row, col)
            neighs = np.array(memo.tolist() + neighs.tolist())
        return neighs

    def get_neighbours(self, input_frame, row, col):
        kernel = self._get_kernel(input_frame, row, col)
        neighs = np.delete(kernel.flatten(), self.remove_idxs)
        neighs = self._memory_handler(row, col, neighs)
        return neighs

    def get_label(self, output_frame, row, col):
        kernel = self._get_kernel(output_frame, row, col)
        label = kernel.flatten()[self.conclusion_idx]
        return label


class KernelD1K9(Kernel):
    """
    Kernel to get neighbouring cells (features).

    This kernel returns 8 nearest neighbours and the cell in the middle (conclusion cell).
    """
    def __init__(self, memory=None):
        super().__init__()
        self.memory = memory
        self.len_memory = self.memory.len_output if self.memory else 0
        self.d = 1
        self.k = 9 + self.len_memory
        self.remove_idxs = []
        self.conclusion_idx = 4
        self.indices = np.arange(self.k).tolist()


class KernelD2K25(Kernel):
    def __init__(self, memory=None):
        super().__init__()
        self.memory = memory
        self.len_memory = self.memory.len_output if self.memory else 0
        self.d = 2
        self.k = 25 + self.len_memory
        self.remove_idxs = []
        self.conclusion_idx = 12
        self.indices = np.arange(self.k).tolist()
