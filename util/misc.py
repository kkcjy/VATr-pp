# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import argparse
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


class EpochLossTracker:
    def __init__(self):
        self.vals = defaultdict(lambda: 0.0)
        self.batch_cnt = 0

    def add_batch(self, losses: dict):
        for k, v in losses.items():
            self.vals[k] += v

        self.batch_cnt += 1

    def get_epoch_loss(self):
        return {k: v / self.batch_cnt for k, v in self.vals.items()}

    def reset(self):
        self.vals = defaultdict(lambda: 0.0)
        self.batch_cnt = 0


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, win_sz=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=win_sz)
        self.total = 0.0
        self.cnt = 0
        self.fmt = fmt

    def update(self, val, n=1):
        self.deque.append(val)
        self.cnt += n
        self.total += val * n

    def sync_procs(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_init():
            return
        t = torch.tensor([self.cnt, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.cnt = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.cnt

    @property
    def max(self):
        return max(self.deque)

    @property
    def val(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            val=self.val)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_sz = get_world_sz()
    if world_sz == 1:
        return [data]

    # serialized to a Tensor
    buf = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buf)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_sz = torch.tensor([tensor.numel()], device="cuda")
    sz_list = [torch.tensor([0], device="cuda") for _ in range(world_sz)]
    dist.all_gather(sz_list, local_sz)
    sz_list = [int(sz.item()) for sz in sz_list]
    max_sz = max(sz_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in sz_list:
        tensor_list.append(torch.empty((max_sz,), dtype=torch.uint8, device="cuda"))
    if local_sz != max_sz:
        pad = torch.empty(size=(max_sz - local_sz,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, pad), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for sz, t in zip(sz_list, tensor_list):
        buf = t.cpu().numpy().tobytes()[:sz]
        data_list.append(pickle.loads(buf))

    return data_list


def reduce_dict(in_dict, avg=True):
    """
    Args:
        in_dict (dict): all the values will be reduced
        avg (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    in_dict, after reduction.
    """
    world_sz = get_world_sz()
    if world_sz < 2:
        return in_dict
    with torch.no_grad():
        names = []
        vals = []
        # sort the keys so that they are consistent across processes
        for k in sorted(in_dict.keys()):
            names.append(k)
            vals.append(in_dict[k])
        vals = torch.stack(vals, dim=0)
        dist.all_reduce(vals)
        if avg:
            vals /= world_sz
        red_dict = {k: v for k, v in zip(names, vals)}
    return red_dict


class MetricLogger:
    def __init__(self, delim="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delim = delim

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delim.join(loss_str)

    def sync_procs(self):
        for meter in self.meters.values():
            meter.sync_procs()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delim.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {mem:.0f}'
            ])
        else:
            log_msg = self.delim.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_sec = iter_time.global_avg * (len(iterable) - i)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_str,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        mem=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_str,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total = time.time() - start
        total_str = str(datetime.timedelta(seconds=int(total)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_str, total / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(cmd):
        return subprocess.check_output(cmd, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    msg = f"sha: {sha}, status: {diff}, branch: {branch}"
    return msg


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_list(batch[0])
    return tuple(batch)


def _max_by_axis(lst):
    # type: (List[List[int]]) -> List[int]
    maxes = lst[0]
    for sub in lst[1:]:
        for idx, item in enumerate(sub):
            maxes[idx] = max(maxes[idx], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, dev):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(dev)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(dev)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_list() instead
            return _onnx_nested_tensor_from_list(tensor_list)

        # TODO make it support different-sized images
        max_sz = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_sz = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_sz
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        dev = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=dev)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=dev)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_list() is an implementation of
# nested_tensor_from_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_sz = []
    for i in range(tensor_list[0].dim()):
        max_sz_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_sz.append(max_sz_i)
    max_sz = tuple(max_sz)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        pad = [(s1 - s2) for s1, s2 in zip(max_sz, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, pad[2], 0, pad[1], 0, pad[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, pad[2], 0, pad[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_dist(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_init():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_sz():
    if not is_dist_avail_init():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_init():
        return 0
    return dist.get_rank()


def is_main_proc():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_proc():
        torch.save(*args, **kwargs)


def init_dist_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_sz = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_sz, rank=args.rank)
    torch.distributed.barrier()
    setup_dist(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    bs = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / bs))
    return res


def interpolate(input, sz=None, scale=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, sz, scale, mode, align_corners
            )

        output_sz = _output_sz(2, input, sz, scale)
        output_sz = list(input.shape[:-2]) + list(output_sz)
        return _new_empty_tensor(input, output_sz)
    else:
        return torchvision.ops.misc.interpolate(input, sz, scale, mode, align_corners)


def add_vatr_args(parser):
    parser.add_argument("--feat_model_path", type=str, default='files/resnet_18_pretrained.pth')
    parser.add_argument("--label_encoder", default='default', type=str)
    parser.add_argument("--save_model_path", default='saved_models', type=str)
    parser.add_argument("--dataset", default='IAM', type=str)
    parser.add_argument("--english_words_path", default='files/english_words.txt', type=str)
    parser.add_argument("--wandb", action='store_true')

    parser.add_argument("--no_writer_loss", action='store_true')
    parser.add_argument("--writer_loss_weight", type=float, default=1.0)
    parser.add_argument("--no_ocr_loss", action='store_true')

    parser.add_argument("--img_height", default=32, type=int)
    parser.add_argument("--resolution", default=16, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_examples", default=15, type=int)
    parser.add_argument("--num_writers", default=339, type=int)

    parser.add_argument("--alphabet",
                        default='Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%',
                        type=str)
    parser.add_argument("--special_alphabet", default='ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω', type=str)
    parser.add_argument("--g_lr", default=0.00005, type=float)
    parser.add_argument("--d_lr", default=0.00001, type=float)
    parser.add_argument("--w_lr", default=0.00005, type=float)
    parser.add_argument("--ocr_lr", default=0.00005, type=float)
    parser.add_argument("--epochs", default=100_000, type=int)

    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=742, type=int)
    parser.add_argument("--num_words", default=3, type=int)
    parser.add_argument("--is_cycle", action="store_true")
    parser.add_argument("--add_noise", default=False, type=bool)
    parser.add_argument("--save_model", default=5, type=int)
    parser.add_argument("--save_model_history", default=500, type=int)
    parser.add_argument("--tag", default='debug', type=str)
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument("--query_input", default='unifont', type=str)

    parser.add_argument("--corpus", default="standard", type=str)
    parser.add_argument("--text-augment-strength", default=0.0, type=float)
    parser.add_argument("--text-aug-type", type=str, default="proportional")
    parser.add_argument("--file-suffix", type=str, default=None)

    parser.add_argument("--augment-ocr", action="store_true")
    parser.add_argument("--d-crop-size", type=int, nargs='*')

    return parser


class FakeArgs:
    feat_model_path = 'files/resnet_18_pretrained.pth'
    label_encoder = 'default'
    save_model_path = 'saved_models'
    dataset = 'IAM'
    english_words_path = 'files/english_words.txt'
    wandb = False
    no_writer_loss = False
    writer_loss_weight = 1.0
    no_ocr_loss = False
    img_height = 32
    resolution = 16
    batch_size = 32
    num_workers = 4
    num_epochs = 100
    lr = 0.0001
    num_examples = 15
    is_kld = False
    tn_hidden_dim = 512
    tn_nheads = 8
    tn_dim_feedforward = 512
    tn_dropout = 0.1
    tn_enc_layers = 3
    tn_dec_layers = 3
    alphabet = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    special_alphabet = 'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'
    query_input = 'unifont'
    projection = 'linear'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(alphabet)
    num_writers = 339  # 339 for IAM, 283 for CVL
    g_lr = 0.00005
    d_lr = 0.00005
    w_lr = 0.00005
    ocr_lr = 0.00005
    add_noise = True
    text_augment_strength = 0.0
    corpus = "standard"
    text_aug_start = 0
    text_aug_warmup = 1
    d_crop_size = None
    arch_blur_size = 0


def get_def_args():
    parser = argparse.ArgumentParser()
    parser = add_vatr_args(parser)

    args = parser.parse_args()
    args.num_writers = 339

    return args


class LinScheduler:
    def __init__(self, param_val: float, start_epoch : int = 0, warmup_epochs: int = 0):
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        self.param_val = param_val

    def get_val(self, epoch):
        if self.start_epoch != 0 and epoch < self.start_epoch:
            return 0.0
        else:
            return min(self.param_val, (max(epoch - self.start_epoch, 1) / max(self.warmup_epochs, 1)) * self.param_val)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scheduler = LinScheduler(0.0, 0, 0)

    v= []
    for i in range(1000):
        v.append(scheduler.get_val(i))

    plt.plot(v)
    plt.show()