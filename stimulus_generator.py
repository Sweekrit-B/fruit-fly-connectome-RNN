import math

import torch
from torch.utils.data import IterableDataset, get_worker_info


def generate_bar_batch(batch_size, T, width=16, direction=1, speed=1.0, bar_width=3, contrast=1.0):
    # a single light bar sweeping across the array at constant velocity, wrapping around the edges
    start_pos = torch.rand(batch_size, 1) * width  # (batch, 1) random starting position
    t = torch.arange(T).float().unsqueeze(0)  # (1, T)
    pos = (start_pos + direction * speed * t) % width  # (batch, T) bar center at each timestep
    positions = torch.arange(width).float()  # (width,)
    pos = pos.unsqueeze(-1)  # (batch, T, 1)
    dist = torch.minimum((positions - pos) % width, (pos - positions) % width)  # (batch, T, width), wrap-around distance
    return (dist < bar_width / 2).float() * contrast


def generate_grating_batch(batch_size, T, width=16, direction=1, temporal_freq=0.15, spatial_freq=0.25, contrast=1.0):
    # a drifting sinusoidal grating
    phase0 = torch.rand(batch_size, 1, 1) * 2 * math.pi
    t = torch.arange(T).float().view(1, T, 1)
    x = torch.arange(width).float().view(1, 1, width)
    phase = 2 * math.pi * spatial_freq * x - direction * 2 * math.pi * temporal_freq * t + phase0
    return contrast * torch.sin(phase)


def generate_static_batch(batch_size, T, width=16, contrast=1.0):
    # a random pattern held fixed across time - a no-motion negative example
    frame = (torch.rand(batch_size, 1, width) > 0.5).float() * contrast
    return frame.expand(-1, T, -1).clone()


def generate_stimulus_batch(batch_size, T=24, width=16):
    """
    Vectorized: builds a whole batch at once by randomly assigning each sample to
    rightward motion, leftward motion, or static, mixing bars and gratings with
    randomized speed/frequency/contrast - no per-sample Python loop. Labels are
    P(rightward) for use with a sigmoid output and BCELoss: 1.0 (rightward),
    0.0 (leftward), or 0.5 (static, i.e. no directional preference).
    """
    kind = torch.randint(0, 3, (batch_size,))  # 0 = bar, 1 = grating, 2 = static
    direction = torch.randint(0, 2, (batch_size,)).float() * 2 - 1  # -1 or 1, one per sample
    contrast = torch.empty(batch_size, 1, 1).uniform_(0.5, 1.0)

    positions = torch.arange(width).float()  # (width,)
    t_row = torch.arange(T).float().view(1, T)  # (1, T)
    t_col = torch.arange(T).float().view(1, T, 1)  # (1, T, 1)
    x = positions.view(1, 1, width)  # (1, 1, width)

    # bar: a light bar sweeping at constant velocity, wrapping around the edges
    speed = torch.empty(batch_size, 1).uniform_(0.5, 2.0)
    bar_width = torch.empty(batch_size, 1, 1).uniform_(2, 4)
    start_pos = torch.rand(batch_size, 1) * width
    bar_pos = (start_pos + direction.view(-1, 1) * speed * t_row) % width  # (batch, T)
    bar_pos = bar_pos.unsqueeze(-1)  # (batch, T, 1)
    dist = torch.minimum((positions - bar_pos) % width, (bar_pos - positions) % width)  # (batch, T, width)
    bar_stim = (dist < bar_width / 2).float() * contrast

    # grating: a drifting sinusoid
    temporal_freq = torch.empty(batch_size, 1, 1).uniform_(0.05, 0.2)
    spatial_freq = torch.empty(batch_size, 1, 1).uniform_(0.15, 0.35)
    phase0 = torch.rand(batch_size, 1, 1) * 2 * math.pi
    phase = 2 * math.pi * spatial_freq * x - direction.view(-1, 1, 1) * 2 * math.pi * temporal_freq * t_col + phase0
    grating_stim = contrast * torch.sin(phase)

    # static: a random pattern held fixed across time - a no-motion negative example
    static_frame = (torch.rand(batch_size, 1, width) > 0.5).float() * contrast
    static_stim = static_frame.expand(-1, T, -1)

    kind_3d = kind.view(-1, 1, 1)
    stimulus = torch.where(kind_3d == 0, bar_stim, torch.where(kind_3d == 1, grating_stim, static_stim))

    directional_labels = (direction == 1).float()  # 1.0 rightward, 0.0 leftward
    labels = torch.where(kind == 2, torch.full((batch_size,), 0.5), directional_labels)

    return stimulus, labels


class SyntheticMotionDataset(IterableDataset):
    """
    Yields whole pre-generated batches instead of one sample at a time, so the
    vectorization in generate_stimulus_batch actually has a full batch to work
    over (a batch_size=1 call has no per-sample loop to remove) and DataLoader
    worker processes each generate independent batches in parallel with training.
    Use with DataLoader(..., batch_size=None) since batches are already formed.
    """
    def __init__(self, num_batches, batch_size, T=24, width=16):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.T = T
        self.width = width

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        worker_info = get_worker_info()
        # split the total batches evenly across DataLoader workers so one epoch
        # still yields num_batches total, not num_batches per worker
        num_batches = self.num_batches if worker_info is None else self.num_batches // worker_info.num_workers
        for _ in range(num_batches):
            yield generate_stimulus_batch(self.batch_size, self.T, self.width)
