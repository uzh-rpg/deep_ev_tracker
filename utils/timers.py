import atexit
import time

import numpy as np
import torch

cuda_timers = {}
timers = {}


class CudaTimer:
    def __init__(self, device: torch.device, timer_name: str = ""):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.device = device
        self.start = None
        self.end = None

    def __enter__(self):
        torch.cuda.synchronize(device=self.device)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        assert self.start is not None
        torch.cuda.synchronize(device=self.device)
        end = time.time()
        cuda_timers[self.timer_name].append(end - self.start)


class CudaTimerWithEvents:
    def __init__(self, device: torch.device, timer_name: str = ""):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.device = device
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record(stream=torch.cuda.current_stream(self.device))
        return self

    def __exit__(self, *args):
        self.end.record(stream=torch.cuda.current_stream(self.device))
        self.end.synchronize()
        # For some reason calling "elapsed_time" uses memory on GPU 0.
        cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end) / 1000)


class TimerDummy:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Timer:
    def __init__(self, timer_name=""):
        self.timer_name = timer_name
        if self.timer_name not in timers:
            timers[self.timer_name] = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        time_diff_s = end - self.start  # measured in seconds
        timers[self.timer_name].append(time_diff_s)


def print_timing_info():
    print("== Timing statistics ==")
    skip_warmup = 2
    for timer_name, timing_values in [*cuda_timers.items(), *timers.items()]:
        if len(timing_values) <= skip_warmup:
            continue
        values = timing_values[skip_warmup:]
        timing_value_s = np.mean(np.array(values))
        timing_value_ms = timing_value_s * 1000
        if timing_value_ms > 1000:
            print("{}: {:.2f} s".format(timer_name, timing_value_s))
        else:
            print("{}: {:.2f} ms".format(timer_name, timing_value_ms))


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
