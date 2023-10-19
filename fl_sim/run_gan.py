import os
import random
import sys
import json
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from opts import get_args

# Utility functions
from utils.utils import top1_accuracy, \
    create_model_dir, init_metrics_meter, extend_metrics_dict, metric_to_dict
from utils.tasks import get_task_elements, get_agg, get_sampling, get_optimizer_init
from utils.logger import Logger
import utils.gan_utils as gan_utils

# Main Modules
from worker_gan import initialize_worker, TorchWorkerGAN, TorchWorkerFedGAN
from server import TorchServer
from simulator import ParallelTrainer


def main(args):
    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    Logger()

    if torch.cuda.device_count():
        cuda_support = True
    else:
        Logger.get().warning('CUDA unsupported!!')
        cuda_support = False

    if args.deterministic:
        if cuda_support:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)

    loader_kwargs = {}
    if not args.device == 'cpu':
        loader_kwargs["num_workers"] = args.num_workers
        loader_kwargs["persistent_workers"] = args.num_workers > 0
    train_loader_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'drop_last': True, **loader_kwargs}

    init_model, _, _, test_batch_size, train_datasets, test_dataset = \
        get_task_elements(args.task, args.test_batch_size, args.data_path)
    global_model = init_model().to(args.device)

    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    Logger.get().info(f"Global model has {count_params(global_model)} parameters.")
    if 'fedgan' in args.task:
        Logger.get().info(f"Submodules:")
        Logger.get().info(f"- contentD -> {count_params(global_model.contentD)} params.")
        Logger.get().info(f"- styleD -> {count_params(global_model.styleD)} params.")
        Logger.get().info(f"- G -> {count_params(global_model.G)} params.")
        Logger.get().info(f"- style_map -> {count_params(global_model.style_map)} params.")
    else:
        Logger.get().info(f"Submodules:")
        Logger.get().info(f"- D -> {count_params(global_model.D)} params.")
        Logger.get().info(f"- G -> {count_params(global_model.G)} params.")

    local_model_s = [init_model().to(args.device) for _ in range(args.simulated_workers)]
    train_loader_s = [DataLoader(dataset, args.batch_size, shuffle=True, **loader_kwargs) for
                      dataset in train_datasets]

    local_opt_init = get_optimizer_init(args.local_opt, args.local_lr)
    global_opt_init = get_optimizer_init(args.global_opt, args.global_lr)
    server_opt = global_opt_init(global_model.parameters())
    agg = get_agg(args.aggregation)
    client_sampler = get_sampling(args.sampling, args.comm_rounds, args.simulated_workers,
                                  len(train_loader_s), args.seed)

    # EDIT YOUR METRICS OF INTEREST HERE
    worker_class = TorchWorkerFedGAN if 'fedgan' in args.task else TorchWorkerGAN
    metrics = {k: (lambda _: 0.0) for k in worker_class.METRICS}
    Logger.get().debug(f"metrics: {list(metrics.keys())}")

    server = TorchServer(
        global_model=global_model,
        optimizer=server_opt
    )
    trainer = ParallelTrainer(
        server=server,
        aggregator=agg,
        client_sampler=client_sampler,
        datasets=train_datasets,
        data_loader_kwargs=train_loader_kwargs,
        log_interval=args.log_interval,
        metrics=metrics,
        device=args.device,
        lr_sched=args.lr_sched,
        aggregate_optim=args.aggregate_optim,
    )

    for worker_id, w_model in enumerate(local_model_s):
        worker = initialize_worker(
            worker_id=worker_id,
            model=w_model,
            optimizer_init=local_opt_init,
            device=args.device,
            server=server,
            log_interval=args.log_interval,
            D_iters=args.D_iters,
            ssl_reg=args.ssl_reg,
            conditional='conditional' in args.task,
            fedgan='fedgan' in args.task,
            linear='linear' in args.task,
        )
        trainer.add_worker(worker)

    full_metrics = init_metrics_meter(metrics)
    model_dir = create_model_dir(args)
    if os.path.exists(os.path.join(
            model_dir, 'full_metrics.json')):
        Logger.get().info(f"{model_dir} already exists.")
        Logger.get().info("Skipping this setup.")
        return
    # create model directory
    os.makedirs(model_dir, exist_ok=True)
    # create sub-directories
    ckpt_dir = os.path.join(model_dir, "model")
    img_dir = os.path.join(model_dir, "imgs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # load models
    if args.load is not None:
        if not os.path.exists(args.load):
            Logger.get().warning(f"Load path '{args.load}' does not exist!")
            Logger.get().warning("Starting from scratch.")
        else:
            Logger.get().info(f"Loading model from {args.load}")
            if 'fedgan' in args.task:
                gan_utils.load_fedgan(trainer, args.load, global_only='unseen' in args.task,
                                      map_location=args.device, reset_optimizers='unseen' in args.task)
            else:
                gan_utils.load_gan(trainer, args.load, map_location=args.device)

    # start training
    for comm_round in range(1, args.comm_rounds + 1):
        Logger.get().info(f"Communication round {comm_round}/{args.comm_rounds}")
        train_metric = trainer.train(comm_round, local_epochs=args.local_epochs)
        extend_metrics_dict(
            full_metrics, metric_to_dict(train_metric, metrics, comm_round, 'train'))

        # save img per worker
        if comm_round % args.eval_every == 0 or comm_round == args.comm_rounds:
            @torch.no_grad()
            def save_snapshot(w):
                img_name = f'w{w.worker_id:02d}_round{comm_round:03d}.png'
                fp = os.path.join(img_dir, img_name)
                Logger.get().info(f"Saving progress snapshot to {fp}")
                tensor = w.progress_frames[-1]
                w.progress_frames = w.progress_frames[-1:]  # XXX
                im = Image.fromarray(gan_utils.tensor_to_np(tensor))
                im.save(fp)

            if 'linear' not in args.task:
                trainer.parallel_call(save_snapshot)

        # save global model and local models per worker
        if comm_round % args.save_every == 0 or comm_round == args.comm_rounds:
            # fp = os.path.join(ckpt_dir, f'model_round{comm_round:03d}.pth.tar')
            fp = os.path.join(ckpt_dir, 'model.pth.tar')  # XXX: add --overwrite flag
            Logger.get().info(f"Saving model to {fp}")
            if 'fedgan' in args.task:
                gan_utils.save_fedgan(trainer, fp)
            else:
                gan_utils.save_gan(trainer, fp)

    #  store the run
    with open(os.path.join(
            model_dir, 'full_metrics.json'), 'w') as f:
        json.dump(full_metrics, f, indent=4)

    # # store progress videos for each worker
    # for w in trainer.workers:
    #     img_name = f'w{w.worker_id:02d}_progress.mp4'
    #     gan_utils.make_animation(w.progress_frames, os.path.join(model_dir, img_name))


if __name__ == "__main__":
    args = get_args(sys.argv)
    main(args)
    torch.cuda.empty_cache()
