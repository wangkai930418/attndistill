import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F

from torch.autograd import Variable
from torch import autograd

def train_one_epoch(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, loss_scaler,
    log_writer=None,
        args=None):
    student_model.train(True)

    eps=1e-5

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    mse_loss = torch.nn.MSELoss().cuda()
    optimizer.zero_grad()
    # ============ some parameters for interpolation
    if not args.distributed:
        student_patch = student_model.patch_size
        teacher_patch = teacher_model.patch_size
        teacher_num_patch_per_dim = int(224/teacher_patch)
        student_num_patch_per_dim = int(224/student_patch)
        teacher_heads = teacher_model.num_heads
    else:
        student_patch = student_model.module.patch_size
        teacher_patch = teacher_model.patch_size
        teacher_num_patch_per_dim = int(224/teacher_patch)
        student_num_patch_per_dim = int(224/student_patch)
        teacher_heads = teacher_model.num_heads

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # with autograd.detect_anomaly():
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        current_batch_length = len(samples)
        identity = torch.eye(current_batch_length).to(device)
        # we use a per iteration (instead of per epoch) lr scheduler
        # remove the adjust learning rate if we are tuning on the base model
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            cls_token_s, attn_s = student_model(samples,)
            if torch.isnan(cls_token_s.max()) or torch.isnan(attn_s.max()):
                del cls_token_s
                del attn_s
                del samples
                continue

            # #########################pred_t is fixed  MASK 0%
            cls_token_t, attn_t = teacher_model(samples)

            # Can also be cosine distance
            assert cls_token_s.shape == cls_token_t.shape

            if args.weight_cls > 0:
                loss_cls = mse_loss(cls_token_s, cls_token_t) * args.weight_cls
            else:
                loss_cls = torch.Tensor([0]).cuda()

            if args.weight_attn > 0:
                # ========== attention aggregation. HYPERPARAMETER:attn_T
                if args.interpolate:
                    # ======= attention interpolation ======
                    attn_t_cls = attn_t[:, :, 0].unsqueeze(-1)
                    attn_t_patch = attn_t[:, :, 1:]

                    attn_t_patch = attn_t_patch.reshape(
                        current_batch_length, teacher_heads, teacher_num_patch_per_dim, teacher_num_patch_per_dim)
                    attn_t_patch = F.interpolate(
                        attn_t_patch, size=student_num_patch_per_dim, mode='bicubic').reshape(
                            current_batch_length, teacher_heads,-1)
                    patch_attn_sum=1-attn_t_cls
                    scale_factor=patch_attn_sum/attn_t_patch.sum(-1).unsqueeze(-1)
                    attn_t_patch = attn_t_patch*scale_factor
                    attn_t = torch.cat((attn_t_cls, attn_t_patch), dim=-1)

                if args.aggregation:
                    attn_s = attn_s.log().sum(dim=1)
                    attn_s = F.softmax((attn_s)/args.attn_T, dim=-1)

                    attn_t = attn_t.log().sum(dim=1)
                    attn_t = F.softmax((attn_t)/args.attn_T, dim=-1)
                
                ## ============= probablities clamp
                attn_s=attn_s.clamp(min=1e-5, max=0.996)
                attn_t=attn_t.clamp(min=1e-5, max=0.996)

                attn_s=F.normalize(attn_s, p=1, dim=-1)
                attn_t=F.normalize(attn_t, p=1,dim=-1)
                
                loss_attn = args.weight_attn * \
                    F.kl_div(attn_s.log(), attn_t, reduction='batchmean')
            else:
                loss_attn = torch.Tensor([0]).cuda()

        with torch.cuda.amp.autocast():
            loss = loss_cls + loss_attn

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(f"cls:{loss_cls.item()}, attn: {loss_attn.item()}")
            print(f"cls_token_s max: {cls_token_s.max()}; cls_token_s min: {cls_token_s.min()}")
            print(f"cls_token_t max: {cls_token_t.max()}; cls_token_t min: {cls_token_t.min()}")

            print(f"attn_s max: {attn_s.max()}; attn_s min: {attn_s.min()}")
            print(f"attn_t max: {attn_t.max()}; attn_t min: {attn_t.min()}")

            print(f"log max: {(attn_t.log()-attn_s.log()).max()}; log min: {(attn_t.log()-attn_s.log()).min()}")
            
            del cls_token_s
            del cls_token_t
            del attn_s
            del attn_t
            del samples
        else:
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=student_model.parameters(),
                            update_grad=(data_iter_step + 1) % accum_iter == 0)

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            # ================================== synchronize ==================================
            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            ################
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(loss_attn=loss_attn.item())

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(
                    (data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}