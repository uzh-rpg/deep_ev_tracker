from pytorch_lightning import Callback


class IncreaseSequenceLengthCallback(Callback):
    def __init__(self, unroll_factor=4, schedule=[40000, 20000, 10000]):
        self.unroll_factor = unroll_factor
        self.schedule = schedule
        self.idx_schedule = 0

    def on_train_batch_end(self, *args):
        if (
            self.idx_schedule < len(self.schedule)
            and args[0].global_step > self.schedule[self.idx_schedule]
        ):
            args[1].unrolls = min(
                args[1].max_unrolls, self.unroll_factor * args[1].unrolls
            )
            self.idx_schedule += 1
            print(
                f"Increasing unrolls: {self.idx_schedule}, {self.schedule[self.idx_schedule]}, {args[0].global_step}"
            )
