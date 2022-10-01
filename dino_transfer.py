from random import choices
from engine.engine_dino_transfer import train_one_epoch
import models.dino_vit_student as dino_vit_student
import models.dino_vit_teacher as dino_vit_teacher

from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
import timm.optim.optim_factory as optim_factory
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

from util.folder2lmdb import ImageFolderLMDB
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm


def get_args_parser():
    parser = argparse.ArgumentParser('vit knowledge distillation', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # teacher model
    parser.add_argument('--teacher_model', default='vit_base_patch16', type=str, metavar='MODEL',
                        choices=dino_vit_teacher.__dict__.keys(),
                        )
    # student model
    parser.add_argument('--student_model', default='vit_small_patch16', type=str, metavar='MODEL',
                        choices=dino_vit_student.__dict__.keys(), )
   
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/datasets/ImageNet/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset_name', default='imagenet', type=str,
                        choices=['imagenet_subset', 'imagenet'])
    parser.add_argument('--save_frequency', default=100, type=int,
                        help='save_frequency')

    parser.add_argument('--output_dir', default='./output_dir/test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/test',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # teacher_resume ===============================
    parser.add_argument('--teacher_resume',
                        default='/data/datasets/SS_ViT/mugs_vit/vit_large_backbone_250ep.pth',
                        help='teacher_resume from checkpoint')
    parser.add_argument('--student_resume', default='',
                        help='student_resume from checkpoint')
    #######################

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--lmdb', action='store_true', default=False,
                        help='whether to use lmdb file')

    ############=========  AttnDistill hyperparameters

    parser.add_argument('--temperature', default=1.0, type=float,
                        help='temperature')
    parser.add_argument('--attn_T', default=20.0, type=float,
                        help='attn_T')
    parser.add_argument('--aggregation', action='store_true', default=False,
                        help='whether to use aggregation')
    parser.add_argument('--interpolate', action='store_true', default=False,
                        help='whether to use aggreinterpolategation')
    ############========= PROJECTOR parameters:
    parser.add_argument('--proj_layers', default=4, type=int,
                        help='proj_layers')
    parser.add_argument('--proj', action='store_true',
                        help='whether proj')
    parser.set_defaults(proj=True)
    parser.add_argument('--noproj', action='store_false', dest='proj',
                        help='whether noproj')

    ############========= hyperparameters for some other methods:
    parser.add_argument('--weight_cls', default=1.0, type=float,
                        help='weight_cls')
    parser.add_argument('--weight_attn', default=20.0, type=float,
                        help='weight_attn')
    ########### distill weights
    parser.add_argument('--dist_type', default='euclidean', type=str, choices=['euclidean', 'cos', ],
                        help='distance type')

    parser.add_argument('--port', default='29501', type=str, choices=['29500', '29501', '29502', '29503'],
                        help='port type')
    
    parser.add_argument('--resume_optim', action='store_true',
                        help='whether resume_optim')
    parser.set_defaults(resume_optim=True)    
    parser.add_argument('--no_resume_optim', action='store_false', dest='resume_optim',
                        help='whether resume_optim') 
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    if 'imagenet' in args.dataset_name:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(
                0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if args.lmdb:
            traindir = os.path.join(args.data_path, 'train.lmdb')
            # valdir = os.path.join(args.data_path, 'val.lmdb')
            dataset_train = ImageFolderLMDB(
                traindir,
                transform_train,
            )
        else:
            dataset_train = datasets.ImageFolder(os.path.join(
                args.data_path, 'train'), transform=transform_train)

        print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
            seed=args.seed + misc.get_rank(),
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=misc.seed_worker
    )

    # define the model model_mae
    teacher_model = dino_vit_teacher.__dict__[args.teacher_model]()

    misc.load_teacher_model(args, args.teacher_resume, teacher_model)
    misc.freeze_model(teacher_model)
    teacher_model.to(device)

    # define the model  models_trans
    student_model = dino_vit_student.__dict__[args.student_model](
        proj_layers=args.proj_layers,
        proj_signal=args.proj,
    )
    student_model.to(device)

    model_without_ddp = student_model
    print("Model = %s" % str(model_without_ddp))

    # set accum_iter
    args.accum_iter = 4096 // (args.batch_size * misc.get_world_size())
    ############
    print(
        f'args accum_iter = {args.accum_iter}, batch size: {args.batch_size}, n_gpus: {misc.get_world_size()}')
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        student_model = torch.nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = student_model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups_mae = optim_factory.add_weight_decay(
        model_without_ddp, args.weight_decay)

    param_groups = param_groups_mae
    # + param_groups_proj

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    ############################################################
    misc.load_student_model(args, args.student_resume,
                            model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            teacher_model,
            student_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_frequency == 0 or epoch + 1 == args.epochs):
            misc.save_transfer_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
