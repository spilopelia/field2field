import torch
import lightning.pytorch as pl
from datapile import FastPMPile, HuggingfaceLoader, CSVHDF5DataModule
from model import Lpt2NbodyNetLightning
import yaml
import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from evaluation import SlicePlotCallback
import os
# Function to load the YAML configuration file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Argument parser to accept the config file path from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Trainer with YAML configuration and WandB integration")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    
    # Adding new arguments
    parser.add_argument('--gpus', type=int, help='Specify the GPU number to use (default: 0)')
    parser.add_argument('--num_nodes', type=int, help='Specify the GPU number to use (default: 0)')
    parser.add_argument('--num_workers', type=int, help='Specify the number of workers (default: 1)')
    parser.add_argument('--train_style_only', action='store_true', help='Only train style_weight and style_bias')
    parser.add_argument('--load_unstyle_model', action='store_true', help='load pretrained unstyle model')

    return parser.parse_args()

# Main function to run the training
def main():

    # Parse command line arguments
    args = parse_args()
    config = load_config(args.config)
    config['model']['batch_size'] = config['data']['batch_size']
    config['model']['max_epochs'] = config['trainer']['max_epochs']
    config['model']['density'] = config['data']['density']
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers  # Update the workers count in the config

    if args.gpus is not None:
        config['trainer']['gpus'] = args.gpus  # Update the GPU count in the config

    if args.num_nodes is not None:
        config['trainer']['num_nodes'] = args.num_nodes  # Update the number of nodes in the config

    config_file_name = os.path.basename(args.config)  # Get the file name
    config_file_name = os.path.splitext(config_file_name)[0]  # Remove the extension
    if args.train_style_only:
        config['model']['update_style_only'] = True
    model = Lpt2NbodyNetLightning(**config['model'])

    if args.train_style_only:
        print("Freezing all parameters except style_weight and style_bias...")
        for name, param in model.named_parameters():
            param.requires_grad = name.endswith("style_weight") or name.endswith("style_bias")

    # Extract data parameters from the config
    # data_module = FastPMPile(**config['data'])
    dataset_type = config['data']['dataset_type']
    if dataset_type == 'raw':
        data_module = FastPMPile(**config['data'])
    elif dataset_type == 'huggingface':
        data_module = HuggingfaceLoader(**config['data']) # faster data pile
    elif dataset_type == 'hdf5':
        data_module = CSVHDF5DataModule(**config['data'])

    # Extract trainer parameters from the config
    gpus = config['trainer']['gpus'] if torch.cuda.is_available() else None
    max_epochs = config['trainer']['max_epochs']
    num_nodes = config['trainer']['num_nodes']
    ckpt_path = config['trainer'].get('ckpt_path', None)
    gradient_clip_val = config['trainer'].get('gradient_clip_val', 0)
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity', None),
        log_model=config['wandb'].get('log_model', False),
        save_dir=config['wandb'].get('save_dir', './wandb_logs'),  # Optional save directory
        id=config['wandb'].get('id', None),
        resume=config['wandb'].get('resume', None),
        name=config['wandb'].get('name', None),
        group=config['wandb'].get('group', None),
    )
    # Get the current WandB run ID
    # Create a checkpoint directory using the WandB run ID
    checkpoint_dir = os.path.join('new_checkpoints', config_file_name)

    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_loss',
        dirpath=checkpoint_dir,
        filename='best-checkpoint-{epoch:02d}',
        save_top_k=1,
        mode='min',
        save_last=True,
        verbose=True
    )

    sliceplot_callback = SlicePlotCallback()

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize the PyTorch Lightning trainer
    strategy = 'ddp' if (gpus is not None and gpus > 1) else 'auto'
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=gpus,
        accelerator="gpu",
        logger=wandb_logger,
        num_nodes=num_nodes,
        strategy=strategy,
        callbacks=[checkpoint_callback, sliceplot_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip_val,
        profiler='simple',  
    )
    if args.load_unstyle_model:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # Filter out incompatible keys
        state_dict = checkpoint["state_dict"]
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape
        }
        model.load_state_dict(filtered_state_dict, strict=False)
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    # Optionally test the model
    # trainer.test(datamodule=data_module)

if __name__ == "__main__":
    main()
