import numpy as np
import collections


class Memory(object):
    def __init__(self):
        self.max_len = None
        self.direction = None
        self.input_original = None
        self.len_output = None

    def init_memory(self, input_original):
        self.input_original = input_original

    def _walk_vertical(self):
        rows = self.input_original.shape[0]
        r = reversed(range(rows)) if self.direction[2:] == "tb" else range(rows)
        for i in r:
            yield i

    def _walk_horizontal(self):
        cols = self.input_original.shape[1]
        r = reversed(range(cols)) if self.direction[:2] == "rl" else range(cols)
        for i in r:
            yield i


class MemoryLTVHR(Memory):
    def __init__(self, max_len=5, direction="tblr"):
        super().__init__()
        self.direction = direction
        self.max_len = max_len
        self.len_output = 2 * self.max_len
        self.dq_h = collections.deque(maxlen=self.max_len)
        self.dq_v = collections.deque(maxlen=self.max_len)

    def reset(self):
        self.dq_h = collections.deque(maxlen=self.max_len)
        self.dq_v = collections.deque(maxlen=self.max_len)

    def _get_comparison_value(self, dq):
        comp = dq[-1] if len(dq) > 0 else -1
        return comp

    def learn(self, row_idx, col_idx):
        for i in self._walk_horizontal():
            v = self.input_original[row_idx, i]
            if int(v) != self._get_comparison_value(self.dq_h):
                self.dq_h.append(int(v))
        for i in self._walk_vertical():
            v = self.input_original[i, col_idx]
            if int(v) != self._get_comparison_value(self.dq_v):
                self.dq_v.append(int(v))
        l0 = list(self.dq_h) + [999] * (self.max_len - len(self.dq_h))
        l1 = list(self.dq_v) + [999] * (self.max_len - len(self.dq_v))
        memory = np.array(l0 + l1)
        self.reset()
        return memory


class MemoryLTVR(Memory):
    def __init__(self, max_len=5, direction="tblr"):
        super().__init__()
        self.direction = direction
        self.max_len = max_len
        self.len_output = self.max_len
        self.dq_v = collections.deque(maxlen=self.max_len)

    def reset(self):
        self.dq_v = collections.deque(maxlen=self.max_len)

    def _get_comparison_value(self, dq):
        comp = dq[-1] if len(dq) > 0 else 0
        return comp

    def learn(self, row_idx, col_idx):
        for i in self._walk_vertical():
            v = self.input_original[i, col_idx]
            if int(v) != self._get_comparison_value(self.dq_v):
                self.dq_v.append(int(v))
        l = list(self.dq_v) + [999] * (self.max_len - len(self.dq_v))
        memory = np.array(l)
        self.reset()
        return memory


class MemoryLTVHR2N(Memory):
    """
    Memory with 2 neighbours up and down.
    """
    def __init__(self, max_len=5, direction="tblr"):
        super().__init__()
        self.direction = direction
        self.max_len = max_len
        self.len_output = 6 * self.max_len
        self.neigh_range = ["u", "m", "d"]
        self.diff_range = {"u": -1, "m": 0, "d": 1}
        self.dq_h, self.dq_v = {}, {}
        for r in self.neigh_range:
            self.dq_h[r] = collections.deque(maxlen=self.max_len)
            self.dq_v[r] = collections.deque(maxlen=self.max_len)

    def reset(self):
        for r in self.neigh_range:
            self.dq_h[r] = collections.deque(maxlen=self.max_len)
            self.dq_v[r] = collections.deque(maxlen=self.max_len)

    def _get_comparison_value(self, dq):
        comp = dq[-1] if len(dq) > 0 else -1
        return comp

    def learn(self, row_idx, col_idx):
        vh, vv = {}, {}
        for i in self._walk_horizontal():
            for r in self.neigh_range:
                try:
                    vh[r] = self.input_original[row_idx+self.diff_range[r], i]
                    if int(vh[r]) != self._get_comparison_value(self.dq_h[r]):
                        self.dq_h[r].append(int(vh[r]))
                except:
                    pass
        for i in self._walk_vertical():
            for r in self.neigh_range:
                try:
                    vv[r] = self.input_original[i, col_idx+self.diff_range[r]]
                    if int(vv[r]) != self._get_comparison_value(self.dq_v[r]):
                        self.dq_v[r].append(int(vv[r]))
                except:
                    pass
        l = []
        for r in self.neigh_range:
            l_ = list(self.dq_h[r]) + [999] * (self.max_len - len(self.dq_h[r]))
            l = l + l_
        for r in self.neigh_range:
            l_ = list(self.dq_v[r]) + [999] * (self.max_len - len(self.dq_v[r]))
            l = l + l_
        memory = np.array(l)
        self.reset()
        return memory
