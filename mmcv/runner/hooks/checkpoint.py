
import os
from ..dist_utils import allreduce_params, master_only
from .hook import HOOKS, Hook


@HOOKS.register_module()
class CheckpointHook(Hook):
    'Save checkpoints periodically.\n\n    Args:\n        interval (int): The saving period. If ``by_epoch=True``, interval\n            indicates epochs, otherwise it indicates iterations.\n            Default: -1, which means "never".\n        by_epoch (bool): Saving checkpoints by epoch or by iteration.\n            Default: True.\n        save_optimizer (bool): Whether to save optimizer state_dict in the\n            checkpoint. It is usually used for resuming experiments.\n            Default: True.\n        out_dir (str, optional): The directory to save checkpoints. If not\n            specified, ``runner.work_dir`` will be used by default.\n        max_keep_ckpts (int, optional): The maximum checkpoints to keep.\n            In some cases we want only the latest few checkpoints and would\n            like to delete old ones to save the disk space.\n            Default: -1, which means unlimited.\n        sync_buffer (bool): Whether to synchronize buffers in different\n            gpus. Default: False.\n    '

    def __init__(self, interval=(- 1), by_epoch=True, save_optimizer=True, out_dir=None, max_keep_ckpts=(- 1), sync_buffer=False, **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.sync_buffer = sync_buffer

    def after_train_epoch(self, runner):
        if ((not self.by_epoch) or (not self.every_n_epochs(runner, self.interval))):
            return
        runner.logger.info(
            ''.join(['Saving checkpoint at ', '{}'.format((runner.epoch + 1)), ' epochs']))
        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner)

    @master_only
    def _save_checkpoint(self, runner):
        'Save the current checkpoint and delete unwanted checkpoint.'
        if (not self.out_dir):
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        if (runner.meta is not None):
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format((runner.epoch + 1))
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format((runner.iter + 1))
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = os.path.join(
                self.out_dir, cur_ckpt_filename)
        if (self.max_keep_ckpts > 0):
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = (runner.epoch + 1)
            else:
                name = 'iter_{}.pth'
                current_ckpt = (runner.iter + 1)
            redundant_ckpts = range(
                (current_ckpt - (self.max_keep_ckpts * self.interval)), 0, (- self.interval))
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = os.path.join(
                    self.out_dir, filename_tmpl.format(_step))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if (self.by_epoch or (not self.every_n_iters(runner, self.interval))):
            return
        runner.logger.info(''.join(
            ['Saving checkpoint at ', '{}'.format((runner.iter + 1)), ' iterations']))
        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner)
