import argparse
import os
import sys
from shutil import move
from datetime import datetime
from contextlib import nullcontext
import random
import numpy as np

import torch
import torch.nn as nn

import mlflow

import torch.nn.functional as F
from torch.utils.data import DataLoader

# tensorboard
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import *
from utils.losses import *
from models.Xception import *
from models.CNNDCT import *
from models.A import *

# for multiprocessing
def setup_for_distributed(is_master):
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

# for removing damaged images
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def parse_args():
    parser = argparse.ArgumentParser()
    
    ## job
    parser.add_argument("--id", type=int, help="unique ID from Slurm")
    parser.add_argument("--run_name", type=str, default="freq", help="run name")

    parser.add_argument("--seed", type=int, default=3721, help="seed")

    ## multiprocessing
    parser.add_argument('--dist_backend', default='nccl', choices=['gloo', 'nccl'], help='multiprocessing backend')
    parser.add_argument('--port', type=int, default=3721, help='port')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    
    ## dataset
    parser.add_argument("--paths_file", type=str, default="/dataset/iut_files.txt", help="path to the file with paths for image under test") # each line of this file should contain "/path/to/image.ext i", i is an integer represents classes
    parser.add_argument("--val_paths_file", type=str, help="path to the validation set")
    parser.add_argument("--n_c_samples", type=int, help="samples per classes (None for non-controlled)")
    parser.add_argument("--val_n_c_samples", type=int, help="samples per classes for validation set (None for non-controlled)")

    parser.add_argument("--image_size", type=int, default=128, help="size of images after resize")

    parser.add_argument("--workers", type=int, default=0, help="number of cpu threads to use during batch generation")

    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches") # note that batch size will be multiplied by n_classes in the end

    ## model
    parser.add_argument('--model', default='xception', choices=['xception', 'cnndct','cnnpixel','ours'], help='model selection')

    parser.add_argument('--load_path', type=str, help='pretrained checkpoint for continued training')

    ## optimizer and scheduler
    parser.add_argument("--optim", choices=['adam', 'adamw'], default='adamw', help="optimizer")

    parser.add_argument('--factor', type=float, default=0.1, help='factor of decay')

    parser.add_argument('--patience', type=int, default=5, help='numbers of epochs to decay for ReduceLROnPlateau scheduler (None to disable)')

    parser.add_argument('--decay_epoch', type=int, help='numbers of epochs to decay for StepLR scheduler (low priority, None to disable)')

    ## training
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")

    parser.add_argument("--cond_epoch", type=int, default=0, help="epoch to start training from")
    
    parser.add_argument("--n_early", type=int, default=10, help="number of epochs for early stopping")

    parser.add_argument("--cond_state", type=str, help="state file for continued training")

    ## log
    parser.add_argument('--save_dir', type=str, default='.', help='dir to save checkpoints and logs')
    parser.add_argument("--log_interval", type=int, default=0, help="interval between saving image samples")
    
    ## DataLad
    parser.add_argument("--repo", type=str, help="repository URL")
    parser.add_argument("--commit", type=str, help="commit hash")
    parser.add_argument("--name", type=str, help="name of the dataset")

    ## MLflow
    parser.add_argument("--workspace", type=str, help="workspace name")
    parser.add_argument("--experiment", type=str, help="experiment name")

    args = parser.parse_args()

    return args

def init_env(args, local_rank, global_rank):
    # for debug only
    #torch.autograd.set_detect_anomaly(True)

    torch.cuda.set_device(local_rank)

    args, checkpoint_dir = init_env_multi(args, global_rank)

    return args, checkpoint_dir

def init_env_multi(args, global_rank):
    setup_for_distributed(global_rank == 0)

    if (args.id is None):
        args.id = datetime.now().strftime("%Y%m%d%H%M%S")

    # set random number
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # checkpoint dir
    checkpoint_dir = args.save_dir + "/checkpoints/" + str(args.id) + "_" + args.run_name
    if global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # finalizing args, print here
    print(args)

    return args, checkpoint_dir

def init_models(args):
    if (args.model == 'xception'):
        model = Xception().cuda()
    elif (args.model == 'cnndct' or args.model == 'cnnpixel'):
        model = CNNDCT(args.image_size).cuda()
    elif (args.model == 'ours'):
        model = Attributor(args.image_size).cuda()
    else:
        print("Unrecognized model %s" % args.model)

    return model

def init_dataset(args, global_rank, world_size, val = False):
    # return None if no validation set provided
    if (val and args.val_paths_file is None):
        print('No val set!')
        return None, None
    # decide if we need to convert input into DCT domain
    if (args.model == 'xception' or args.model == 'cnnpixel'):
        dct = False
    elif (args.model == 'cnndct' or args.model == 'ours'):
        dct = True
    else:
        print("Unrecognized model %s" % args.model)

    dataset = AnimeDataset(global_rank,
                                (args.paths_file if not val else args.val_paths_file),
                                args.id,
                                args.image_size,
                                dct,
                                (args.n_c_samples if not val else args.val_n_c_samples),
                                val,
                                args.repo, args.commit, args.name)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    local_batch_size = args.batch_size // world_size

    if (not val):
        print('Local batch size is {} ({}//{})!'.format(local_batch_size, args.batch_size, world_size))

    dataloader = DataLoader(dataset=dataset, batch_size=local_batch_size, num_workers=args.workers, pin_memory=True, drop_last=True, sampler=sampler, collate_fn=collate_fn)

    n_drop = len(dataloader.dataset) - len(dataloader) * args.batch_size
    print('{} set size is {} (drop_last {})!'.format(('Train' if not val else 'Val'), len(dataloader) * args.batch_size, n_drop))

    return sampler, dataloader

def init_optims(args, world_size,
                model):
    
    # Optimizers
    local_lr = args.lr / world_size

    print('Local learning rate is %.3e (%.3e/%d)!' % (local_lr, args.lr, world_size))

    if (args.optim == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=local_lr)
    elif (args.optim == 'adamw'):
        optimizer = torch.optim.AdamW(model.parameters(), lr=local_lr)
    else:
        print("Unrecognized optimizer %s" % args.optim)
        sys.exit()

    print("Using optimizer {}".format(args.optim))

    return optimizer

def init_schedulers(args, optimizer):
    lr_scheduler = None

    # high priority for ReduceLROnPlateau (validation set required)
    if (args.val_paths_file and args.patience):
        print("Using scheduler ReduceLROnPlateau")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                    factor = args.factor,
                                                    patience = args.patience)
    # low priority StepLR
    elif (args.decay_epoch):
        print("Using scheduler StepLR")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
                                                    step_size = args.decay_epoch,
                                                    gamma = args.factor)
    
    else:
        print("No scheduler used")

    return lr_scheduler

def load_dicts(args,
                model):
    # Load pretrained models
    if args.load_path != None and args.load_path != 'timm':
        print('Load pretrained model: {}'.format(args.load_path))

        model.load_state_dict(torch.load(args.load_path))

    return model

# for saving checkpoints
def save_checkpoints(args, checkpoint_dir, id, epoch, save_best, last_best, get_module,
                    model):
    if (get_module):
        net = model.module
    else:
        net = model

    # always remove save last and remove previous
    torch.save(net.state_dict(),
                os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch) + '.pth'))

    last_pth = os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch - 1) + '.pth')
    if (os.path.exists(last_pth)):
        os.remove(last_pth)

    # save best
    save_best_pth = None
    if (save_best):
        save_best_pth = os.path.join(checkpoint_dir, str(id) + "_best_" + str(epoch) + '.pth')
        torch.save(net.state_dict(), save_best_pth)

        # last_best_pth = os.path.join(checkpoint_dir, str(id) + "_best_" + str(last_best) + '.pth')
        # if (os.path.exists(last_best_pth)):
        #     os.remove(last_best_pth)

        print('Checkpoint saved to %s.' % (save_best_pth))

    return last_pth, save_best_pth

def predict_loss(args, data, model,
                criterion_BCE):
    # load data
    in_iuts, in_labels = data

    in_iuts = in_iuts.to('cuda', non_blocking=True)
    in_labels = in_labels.to('cuda', non_blocking=True)

    # predict
    out_labels = model(in_iuts)
    
    # calculate loss
    loss = criterion_BCE(out_labels, in_labels.float().unsqueeze(1))

    return loss

def reset_optim_lr(optimizer, lr, world_size):
    local_lr = lr / world_size

    for g in optimizer.param_groups:
        g['lr'] = local_lr

    return optimizer

def save_state(checkpoint_dir, id, epoch, save_best, last_best,
                optimizer, lr_scheduler,
                best_val_loss, n_last_epochs):
    
    state = {'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
             'best_val_loss': best_val_loss, 'n_last_epochs': n_last_epochs}

    # save last
    torch.save(state,
                os.path.join(checkpoint_dir, 'state_' + str(id) + "_last_" + str(epoch) + '.pth'))

    last_pth = os.path.join(checkpoint_dir, 'state_' + str(id) + "_last_" + str(epoch - 1) + '.pth')
    if (os.path.exists(last_pth)):
        os.remove(last_pth)

    # save best
    if (save_best):
        save_best_pth = os.path.join(checkpoint_dir, 'state_' + str(id) + "_best_" + str(epoch) + '.pth')
        torch.save(state, save_best_pth)

        # last_best_pth = os.path.join(checkpoint_dir, 'state_' + str(id) + "_best_" + str(last_best) + '.pth')
        # if (os.path.exists(last_best_pth)):
        #     os.remove(last_best_pth)

        print('State saved to %s with best val loss %.3e @%d.' % (save_best_pth, best_val_loss, n_last_epochs))

def train(args, global_rank, world_size, sync, get_module,
            checkpoint_dir,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler,
            prev_best_val_loss, prev_n_last_epochs):
    # Losses that are built-in in PyTorch
    criterion_BCE = nn.BCEWithLogitsLoss().cuda()

    # tensorboard
    if global_rank == 0:
        os.makedirs(args.save_dir + "/logs/", exist_ok=True)
        writer = SummaryWriter(args.save_dir + "/logs/" + str(args.id) + "_" + args.run_name)

    last_best = -1
    new_best = -1
    n_last_epochs = prev_n_last_epochs

    lr_last_best = -1

    if (prev_best_val_loss is not None):
        best_val_loss = prev_best_val_loss

        print("Best val loss set to %.3e" % (prev_best_val_loss))
    else:
        best_val_loss = float('inf')

    # for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = args.cond_epoch
    for epoch in range(start_epoch, args.n_epochs):

        train_sampler.set_epoch(epoch)
        
        print('Starting Epoch {}'.format(epoch))

        # loss sum for epoch
        epoch_loss = 0

        epoch_val_loss = 0

        # ------------------
        #  Train step
        # ------------------
        with model.join() if get_module else nullcontext(): # get_module indicates using DDP         
            for step, data in enumerate(dataloader):
                curr_steps = epoch * len(dataloader) + step

                model.train()

                if (sync): optimizer.synchronize()
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    loss = predict_loss(args, data, model, criterion_BCE)

                # backward prop
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # log losses for epoch
                epoch_loss += loss

                # --------------
                #  Log Progress (for certain steps)
                # --------------
                if args.log_interval != 0 and step % args.log_interval == 0 and global_rank == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                            f"[Epoch {epoch}, Batch {step}/{len(dataloader) - 1}]"
                            f"[Loss {loss:.3e}]")

                    writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], curr_steps)

                    writer.add_scalar("Loss/Train", loss, curr_steps)

        # ------------------
        #  Validation
        # ------------------
        if (val_sampler and val_dataloader):

            val_sampler.set_epoch(epoch)

            model.eval()

            for step, data in enumerate(val_dataloader):
                with torch.no_grad():
                    loss = predict_loss(args, data, model, criterion_BCE)

                    epoch_val_loss += loss.item()

            if epoch_val_loss <= best_val_loss:
                best_val_loss = epoch_val_loss
                n_last_epochs = 0

                new_best = epoch
            else:
                n_last_epochs += 1
        else:
            new_best = epoch

        # ------------------
        #  Step
        # ------------------
        lr_before_step = optimizer.param_groups[0]['lr']

        if (lr_scheduler):
            if (args.val_paths_file and args.patience):
                lr_scheduler.step(epoch_val_loss) # ReduceLROnPlateau
            elif (args.decay_epoch):
                lr_scheduler.step() # StepLR
            else:
                print("Error in scheduler step")
                sys.exit()

        # --------------
        #  Log Progress (for epoch)
        # --------------
        # loss average for epoch
        if (global_rank == 0):
            epoch_loss_avg = epoch_loss / len(dataloader)

            if (val_dataloader):
                epoch_val_loss_avg = epoch_val_loss / len(val_dataloader)
                best_val_loss_avg = best_val_loss / len(val_dataloader)
                lr_best_val_loss_avg = lr_scheduler.best / len(val_dataloader)
            else:
                epoch_val_loss_avg = 0
                best_val_loss_avg = 0

            # global lr (use before-step lr)
            global_lr = lr_before_step * world_size

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                  f"[Epoch {epoch}/{args.n_epochs - 1}]"
                  f"[Loss {epoch_loss_avg:.3e}]"
                  f"[Val Loss {epoch_val_loss_avg:.3e} (Best {best_val_loss_avg:.3e} @{n_last_epochs:d}, LR Best {lr_best_val_loss_avg:.3e} @{lr_scheduler.num_bad_epochs:d})]"
                  f"[LR {global_lr:.3e}]")

            writer.add_scalar("Epoch LearningRate", global_lr, epoch)

            writer.add_scalar("Epoch Loss/Train", epoch_loss_avg, epoch)
            writer.add_scalar("Epoch Loss/Val", epoch_val_loss_avg, epoch)

            # save model parameters
            if global_rank == 0:
                last_pth, save_best_pth = save_checkpoints(args, checkpoint_dir, args.id, epoch,
                                 new_best == epoch, last_best,
                                 get_module,
                                 model)
                
                if (new_best == epoch):
                        last_best = epoch

            # save state
            if global_rank == 0:
                save_state(checkpoint_dir, args.id, epoch, 
                lr_scheduler.num_bad_epochs == 0, lr_last_best,
                 optimizer, lr_scheduler,
                 best_val_loss, n_last_epochs)

                if lr_scheduler.num_bad_epochs == 0:
                    lr_last_best = epoch

            # log to MLflow
            metrics = {
                "train_loss": float(epoch_loss_avg),
                "learning_rate": float(global_lr),
            }
            if (val_dataloader):
                metrics.update({
                    "val_loss": float(epoch_val_loss_avg),
                    "best_val_loss": float(best_val_loss_avg),
                })

            if mlflow.active_run() is not None:
                mlflow.log_metrics(metrics, step=epoch)

                if save_best_pth is not None:
                    mlflow.log_artifact(save_best_pth, "best_models")
                    
        # reset early stopping when learning rate changed
        lr_after_step = optimizer.param_groups[0]['lr']
        if (lr_after_step != lr_before_step):
            print("LR changed to %.3e" % (lr_after_step * world_size))

        # check early_stopping
        if (n_last_epochs > args.n_early or lr_scheduler.num_bad_epochs > args.n_early):
            print('Early stopping')
            break

    # log to MLflow
    if global_rank == 0 and mlflow.active_run() is not None:
        net = model.module if get_module else model
        net.eval()

        mlflow.pytorch.log_model(
            pytorch_model=net,
            name=f"model-{args.run_name}-{args.id}",
            serialization_format="pickle",
        )

    print('Finished training')

    if global_rank == 0:
        writer.close()

    pass