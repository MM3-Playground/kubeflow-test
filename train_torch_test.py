import os
from datetime import timedelta

import torch
import torch.nn as nn

import mlflow

# multiprocessing
from utils.dist import *

from train_base_test import *

# constants
SYNC = False
GET_MODULE = True

def main():
    args = parse_args()

    # Init dist
    init_dist('slurm', args.port)

    global_rank, world_size = get_dist_info()

    args, checkpoint_dir = init_env_multi(args, global_rank)

    # MLflow
    mlflow_run = None
    
    if global_rank == 0:
        mlflow.set_workspace(args.workspace) 
        mlflow.set_experiment(args.experiment)

        mlflow_run = mlflow.start_run(run_name=f"train-{args.run_name}-{args.id}")

        mlflow.log_params({
            "run_id": args.id,
            "run_name": args.run_name,
            "seed": args.seed,
            "model": args.model,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "optimizer": args.optim,
            "learning_rate": args.lr,
            "epochs": args.n_epochs,
            "early_stopping_patience": args.n_early,
            "lr_scheduler_patience": args.patience,
            "lr_decay_factor": args.factor,
            "lr_decay_epoch": args.decay_epoch,
            "world_size": world_size,
            "train_paths_file": args.paths_file,
            "val_paths_file": args.val_paths_file or "",
            "load_path": args.load_path or "",
        })

        mlflow.log_artifact(args.paths_file, "datasets")
        mlflow.log_artifact(args.val_paths_file, "datasets")

    # models
    model = init_models(args)
    model = load_dicts(args, model)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    optimizer = init_optims(args, world_size, model)

    lr_scheduler = init_schedulers(args, optimizer)

    if (args.cond_state is not None):
        saved_state = torch.load(args.cond_state)

        optimizer.load_state_dict(saved_state['optimizer'])
        lr_scheduler.load_state_dict(saved_state['lr_scheduler'])

        prev_best_val_loss = saved_state['best_val_loss']
        prev_n_last_epochs = saved_state['n_last_epochs']

        # move to device
        device = torch.device('cuda:' + str(torch.cuda.current_device()))

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        prev_best_val_loss = None
        prev_n_last_epochs = 0

    # dataset
    train_sampler, dataloader = init_dataset(args, global_rank, world_size, False)
    val_sampler, val_dataloader = init_dataset(args, global_rank, world_size, True)

    train(args, global_rank, world_size, SYNC, GET_MODULE,
            checkpoint_dir,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler,
            prev_best_val_loss, prev_n_last_epochs)

    if global_rank == 0 and mlflow_run is not None:
        mlflow.end_run()

if __name__ == '__main__':
    main()