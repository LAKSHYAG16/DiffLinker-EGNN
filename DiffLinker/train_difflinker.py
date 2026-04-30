import argparse
import getpass
import os
import sys
import yaml

from datetime import datetime
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from src.const import NUMBER_OF_ATOM_TYPES, GEOM_NUMBER_OF_ATOM_TYPES
from src.lightning import DDPM
from src.utils import disable_rdkit_logging, set_deterministic, Logger


def find_last_checkpoint(checkpoints_dir):
    ckpts = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    if len(ckpts) == 0:
        return None
    ckpts = sorted(ckpts)
    return os.path.join(checkpoints_dir, ckpts[-1])


def main(args):
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S')
    username = getpass.getuser()

    run_name = f'{os.path.splitext(os.path.basename(args.config))[0]}_{username}_{args.exp_name}_bs{args.batch_size}_{start_time}'
    experiment = run_name if args.resume is None else args.resume

    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    logs_dir = os.path.join(args.logs, "general_logs", experiment)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    sys.stdout = Logger(logpath=os.path.join(logs_dir, 'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(logs_dir, 'log.log'), syspart=sys.stderr)

    samples_dir = os.path.join(args.logs, 'samples', experiment)

    set_deterministic(args.seed)
    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'

    # Logger
    wandb_logger = None if args.no_wandb else loggers.WandbLogger(
        save_dir=args.logs,
        project='e3_ddpm_linker_design',
        name=experiment,
        id=experiment,
        resume='allow',
        entity=args.wandb_entity,
    )

    # Dataset info
    is_geom = ('geom' in args.train_data_prefix) or ('MOAD' in args.train_data_prefix)
    number_of_atoms = GEOM_NUMBER_OF_ATOM_TYPES if is_geom else NUMBER_OF_ATOM_TYPES

    in_node_nf = number_of_atoms + args.include_charges
    anchors_context = not args.remove_anchors_context
    context_node_nf = 2 if anchors_context else 1
    if '.' in args.train_data_prefix:
        context_node_nf += 1

    # Model
    ddpm = DDPM(
        data_path=args.data,
        train_data_prefix=args.train_data_prefix,
        val_data_prefix=args.val_data_prefix,
        in_node_nf=in_node_nf,
        n_dims=3,
        context_node_nf=context_node_nf,
        hidden_nf=args.nf,
        activation=args.activation,
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        diffusion_steps=args.diffusion_steps,
        diffusion_noise_schedule=args.diffusion_noise_schedule,
        diffusion_noise_precision=args.diffusion_noise_precision,
        diffusion_loss_type=args.diffusion_loss_type,
        normalize_factors=args.normalize_factors,
        include_charges=args.include_charges,
        lr=args.lr,
        batch_size=args.batch_size,
        torch_device=torch_device,
        model=args.model,
        test_epochs=args.test_epochs,
        n_stability_samples=args.n_stability_samples,
        normalization=args.normalization,
        log_iterations=args.log_iterations,
        samples_dir=samples_dir,
        data_augmentation=args.data_augmentation,
        center_of_mass=args.center_of_mass,
        inpainting=args.inpainting,
        anchors_context=anchors_context,
        graph_type=args.graph_type,
    )

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="best-{epoch:02d}",
        monitor=None,         
        save_top_k=-1,        
        save_last=True
    )

    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator=args.device,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    last_checkpoint = find_last_checkpoint(checkpoints_dir) if args.resume else None

    print('Start training')
    trainer.fit(model=ddpm, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DiffLinker Training')

    # Basic
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/zinc_difflinker.yml')
    p.add_argument('--data', type=str, default="./data/zinc")
    p.add_argument('--train_data_prefix', type=str, default='train')
    p.add_argument('--val_data_prefix', type=str, default='val')
    p.add_argument('--checkpoints', type=str, default='checkpoints')
    p.add_argument('--logs', type=str, default='logs')
    p.add_argument('--device', type=str, default='gpu')
    p.add_argument('--exp_name', type=str, default='Lakshya_Model')

    # Training
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_workers', type=int, default=4)

    # Diffusion
    p.add_argument('--diffusion_steps', type=int, default=500)
    p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2')
    p.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
    p.add_argument('--diffusion_loss_type', type=str, default='l2')

    # Model
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--nf', type=int, default=32)
    p.add_argument('--activation', type=str, default='silu')

    # Required (FIXED ERROR)
    p.add_argument('--inpainting', action='store_true', default=False)
    p.add_argument('--remove_anchors_context', action='store_true', default=False)
    p.add_argument('--graph_type', type=str, default='FC')
    p.add_argument('--normalization', type=str, default='batch_norm')
    p.add_argument('--aggregation_method', type=str, default='sum')
    p.add_argument('--normalization_factor', type=float, default=1)
    p.add_argument('--sin_embedding', type=eval, default=False)
    p.add_argument('--attention', type=eval, default=True)
    p.add_argument('--tanh', type=eval, default=True)
    p.add_argument('--norm_constant', type=float, default=1)
    p.add_argument('--inv_sublayers', type=int, default=1)
    p.add_argument('--include_charges', type=eval, default=True)
    p.add_argument('--data_augmentation', type=eval, default=False)
    p.add_argument('--center_of_mass', type=str, default='fragments')
    p.add_argument('--test_epochs', type=int, default=1)
    p.add_argument('--n_stability_samples', type=int, default=500)
    p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1])
    p.add_argument('--log_iterations', type=int, default=20)

    # Other
    p.add_argument('--wandb_entity', type=str, default='geometric')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_wandb', action='store_true')

    disable_rdkit_logging()

    args = p.parse_args()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            arg_dict[key] = value
        args.config = args.config.name

    main(args=args)