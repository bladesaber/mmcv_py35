
import os.path as osp
import platform
import shutil
import time
import warnings
import torch
from torch.optim import Optimizer
import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .hooks import IterTimerHook
from .utils import get_host_info


class IterLoader():

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    'Iteration-based Runner.\n\n    This runner train models iteration by iteration.\n    '

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if (not isinstance(outputs, dict)):
            raise TypeError('model.train_step() must return a dict')
        if ('log_vars' in outputs):
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if (not isinstance(outputs, dict)):
            raise TypeError('model.val_step() must return a dict')
        if ('log_vars' in outputs):
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        self._inner_iter += 1

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        "Start running.\n\n        Args:\n            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training\n                and validation.\n            workflow (list[tuple]): A list of (phase, iters) to specify the\n                running order and iterations. E.g, [('train', 10000),\n                ('val', 1000)] means running 10000 iterations for training and\n                1000 iterations for validation, iteratively.\n        "
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert (len(data_loaders) == len(workflow))
        if (max_iters is not None):
            warnings.warn(
                'setting max_iters in run is deprecated, please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert (
            self._max_iters is not None), 'max_iters must be specified during instantiation'
        work_dir = (self.work_dir if (self.work_dir is not None) else 'NONE')
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters',
                         workflow, self._max_iters)
        self.call_hook('before_run')
        iter_loaders = [IterLoader(x) for x in data_loaders]
        self.call_hook('before_epoch')
        while (self.iter < self._max_iters):
            for (i, flow) in enumerate(workflow):
                self._inner_iter = 0
                (mode, iters) = flow
                if ((not isinstance(mode, str)) or (not hasattr(self, mode))):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if ((mode == 'train') and (self.iter >= self._max_iters)):
                        break
                    iter_runner(iter_loaders[i], **kwargs)
        time.sleep(1)
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        "Resume model from checkpoint.\n\n        Args:\n            checkpoint (str): Checkpoint to resume from.\n            resume_optimizer (bool, optional): Whether resume the optimizer(s)\n                if the checkpoint file includes optimizer(s). Default to True.\n            map_location (str, optional): Same as :func:`torch.load`.\n                Default to 'default'.\n        "
        if (map_location == 'default'):
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(checkpoint, map_location=(
                lambda storage, loc: storage.cuda(device_id)))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)
        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if (('optimizer' in checkpoint) and resume_optimizer):
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(''.join(
                    ['Optimizer should be dict or torch.optim.Optimizer but got ', '{}'.format(type(self.optimizer))]))
        self.logger.info(''.join(['resumed from epoch: ', '{}'.format(
            self.epoch), ', iter ', '{}'.format(self.iter)]))

    def save_checkpoint(self, out_dir, filename_tmpl='iter_{}.pth', meta=None, save_optimizer=True, create_symlink=True):
        "Save checkpoint to file.\n\n        Args:\n            out_dir (str): Directory to save checkpoint files.\n            filename_tmpl (str, optional): Checkpoint file template.\n                Defaults to 'iter_{}.pth'.\n            meta (dict, optional): Metadata to be saved in checkpoint.\n                Defaults to None.\n            save_optimizer (bool, optional): Whether save optimizer.\n                Defaults to True.\n            create_symlink (bool, optional): Whether create symlink to the\n                latest checkpoint file. Defaults to True.\n        "
        if (meta is None):
            meta = dict(iter=(self.iter + 1), epoch=(self.epoch + 1))
        elif isinstance(meta, dict):
            meta.update(iter=(self.iter + 1), epoch=(self.epoch + 1))
        else:
            raise TypeError(
                ''.join(['meta should be a dict or None, but got ', '{}'.format(type(meta))]))
        if (self.meta is not None):
            meta.update(self.meta)
        filename = filename_tmpl.format((self.iter + 1))
        filepath = osp.join(out_dir, filename)
        optimizer = (self.optimizer if save_optimizer else None)
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if (platform.system() != 'Windows'):
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filename, dst_file)

    def register_training_hooks(self, lr_config, optimizer_config=None, checkpoint_config=None, log_config=None, momentum_config=None):
        'Register default hooks for iter-based training.\n\n        Default hooks include:\n\n        - LrUpdaterHook\n        - MomentumUpdaterHook\n        - OptimizerStepperHook\n        - CheckpointSaverHook\n        - IterTimerHook\n        - LoggerHook(s)\n        '
        if (checkpoint_config is not None):
            checkpoint_config.setdefault('by_epoch', False)
        if (lr_config is not None):
            lr_config.setdefault('by_epoch', False)
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        if (log_config is not None):
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        self.register_logger_hooks(log_config)
