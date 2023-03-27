import torch
import numpy as np
from enum import Enum, auto

import hdf5plugin
import h5py


class EventRepresentationTypes(Enum):
    time_surface = 0
    voxel_grid = 1
    event_stack = 2


class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError


class TimeSurface(EventRepresentation):
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.input_size = input_size
        self.time_surface = torch.zeros(input_size, dtype=torch.float, requires_grad=False)
        self.n_bins = input_size[0] // 2

    def convert(self, events):
        _, H, W = self.time_surface.shape
        with torch.no_grad():
            self.time_surface = torch.zeros(self.input_size, dtype=torch.float, requires_grad=False,
                                            device=events['p'].device)
            time_surface = self.time_surface.clone()

            t = events['t'].cpu().numpy()
            dt_bin = 1. / self.n_bins
            x0 = events['x'].int()
            y0 = events['y'].int()
            p0 = events['p'].int()
            t0 = events['t']

            # iterate over bins
            for i_bin in range(self.n_bins):
                t0_bin = i_bin * dt_bin
                t1_bin = t0_bin + dt_bin

                # mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                # x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
                idx0 = np.searchsorted(t, t0_bin, side='left')
                idx1 = np.searchsorted(t, t1_bin, side='right')
                x_bin = x0[idx0:idx1]
                y_bin = y0[idx0:idx1]
                p_bin = p0[idx0:idx1]
                t_bin = t0[idx0:idx1]

                n_events = len(x_bin)
                for i in range(n_events):
                    if 0 <= x_bin[i] < W and 0 <= y_bin[i] < H:
                        time_surface[2*i_bin+p_bin[i], y_bin[i], x_bin[i]] = t_bin[i]

        return time_surface


class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.input_size = input_size
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = torch.zeros((self.input_size), dtype=torch.float, requires_grad=False,
                                          device=events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            value = 2*events['p']-1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-events['x']).abs()) * (1 - (ylim-events['y']).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


class EventStack(EventRepresentation):
    def __init__(self, input_size: tuple):
        """
        :param input_size: (C, H, W)
        """
        assert len(input_size) == 3
        self.input_size = input_size
        self.event_stack = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert(self, events):
        C, H, W = self.event_stack.shape
        with torch.no_grad():
            self.event_stack = torch.zeros((self.input_size), dtype=torch.float, requires_grad=False,
                                           device=events['p'].device)
            event_stack = self.event_stack.clone()

            t = events['t'].cpu().numpy()
            dt_bin = 1. / self.nb_channels
            x0 = events['x'].int()
            y0 = events['y'].int()
            p0 = 2*events['p'].int()-1
            t0 = events['t']

            # iterate over bins
            for i_bin in range(self.nb_channels):
                t0_bin = i_bin * dt_bin
                t1_bin = t0_bin + dt_bin

                # mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                # x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
                idx0 = np.searchsorted(t, t0_bin, side='left')
                idx1 = np.searchsorted(t, t1_bin, side='right')
                x_bin = x0[idx0:idx1]
                y_bin = y0[idx0:idx1]
                p_bin = p0[idx0:idx1]

                n_events = len(x_bin)
                for i in range(n_events):
                    if 0 <= x_bin[i] < W and 0 <= y_bin[i] < H:
                        event_stack[i_bin, y_bin[i], x_bin[i]] += p_bin[i]

        return event_stack


def events_to_time_surface(time_surface, p, t, x, y):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
    }
    return time_surface.convert(event_data_torch)

def events_to_event_stack(event_stack, p, t, x, y):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
    }
    return event_stack.convert(event_data_torch)


def events_to_voxel_grid(voxel_grid, p, t, x, y):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    event_data_torch = {
        'p': torch.from_numpy(pol),
        't': torch.from_numpy(t),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
    }
    return voxel_grid.convert(event_data_torch)
