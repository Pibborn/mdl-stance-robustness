from torch.utils.tensorboard import SummaryWriter
from data_utils.label_map import TaskIDMap


class TensorboardHandler:

    def __init__(self, path):
        self.writer = SummaryWriter(path)
        self.step_task_counter = {i: 0 for i in list(range(len(TaskIDMap.task_map)))}
        self.step = 0

    def update_tb(self, task_id, value, debias=False, mmd=False, update=False):
        task_name = TaskIDMap.get_name(task_id)
        if debias:
            self.writer.add_scalar('debias_loss', value, self.step)
        else:
            if mmd:
                self.writer.add_scalar('task_{}/mmd_loss'.format(task_name), value, self.step_task_counter[task_id])
                self.writer.add_scalar('mmd_loss', value, self.step)
            else:
                self.writer.add_scalar('task_{}/loss'.format(task_name), value, self.step_task_counter[task_id])
        if update:
            self.step_task_counter[task_id] += 1
            self.step += 1
