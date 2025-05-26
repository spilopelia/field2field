import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from lag2eul import lag2eul
from unet import UNet3D, UNet3DwithRes, UNet3Dwithnopadding, StyledUNet3D, FiLMUNet3D

from torch_ema import ExponentialMovingAverage

from naf_net import NAFNet3D, BaselineBlock, BaselineBlock_SCA, BaselineBlock_SG, NAFBlock3D
from torchmetrics.image import PeakSignalNoiseRatio
import os
torch.backends.cudnn.benchmark = True
    
# LightningModule wrapping the Lpt2NbodyNet
class Lpt2NbodyNetLightning(pl.LightningModule):
    def __init__(self, 
                lr: float = 1.0e-4,
                beta1: float = 0.9,
                beta2: float = 0.999,
                weight_decay: float = 1.0e-4,  
                optimizer: str = 'Adam',
                lr_scheduler: str = 'Constant',
                lr_warmup: int = 1000,
                lr_cosine_period = None,
                num_samples: int = 30000,
                batch_size: int = 128,
                max_epochs: int = 500,
                model: str = 'default',
                num_layers: int = 4, 
                base_filters: int = 64, 
                blocks_per_layer: int = 2, 
                init_dim: int = 3,
                style_size: int = None,
                reversed: bool = False,
                compressed: bool = False,
                compression_type: str = 'arcsinh',
                compression_factor: float = 24,
                eul_loss: bool = False,
                eul_loss_scale: float = 1.0,
                lag_loss_scale: float = 1.0,
                num_input_channels: int = 3,  
                num_output_channels: int = 3 ,  
                ema: bool = False,
                ema_rate: float = 0.999,
                naf_middle_blk_num = 12,
                naf_enc_blk_nums = [2, 2, 4, 8],
                naf_dec_blk_nums = [2, 2, 2, 2],
                naf_dw_expand = 2,
                naf_ffn_expand = 2,
                **kwargs
                ):
        super(Lpt2NbodyNetLightning, self).__init__()

        self.save_hyperparameters(ignore=['kwargs'])  # This will save all init args except kwargs
        self.model_type = model

        if model == "UNet":
            self.model = UNet3D(num_layers=self.hparams.num_layers,
                                base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer,init_dim=self.hparams.init_dim)
        elif model == "UNet3DwithRes":
            self.model = UNet3DwithRes(num_layers=self.hparams.num_layers,
                                base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer,init_dim=self.hparams.init_dim)
        elif model == "UNet3Dwithnopadding":
            self.model = UNet3Dwithnopadding(num_layers=self.hparams.num_layers,
                                base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer,init_dim=self.hparams.init_dim)
        elif model == "StyledUNet3D":
            self.model = StyledUNet3D(num_layers=self.hparams.num_layers,
                                base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer,init_dim=self.hparams.init_dim, style_size = self.hparams.style_size)
        elif model == "FiLMUNet3D":
            self.model = FiLMUNet3D(num_layers=self.hparams.num_layers,
                                base_filters=self.hparams.base_filters,blocks_per_layer=self.hparams.blocks_per_layer,init_dim=self.hparams.init_dim, style_size = self.hparams.style_size)    
        elif model == "NAFNet3D_base":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = BaselineBlock)
        elif model == "NAFNet3D_base_SG":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = BaselineBlock_SG)
        elif model == "NAFNet3D_base_SCA":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = BaselineBlock_SCA)
        elif model == "NAFNet3D":
                self.model = NAFNet3D(img_channel=self.hparams.init_dim, width=self.hparams.base_filters, middle_blk_num=self.hparams.naf_middle_blk_num, 
                                enc_blk_nums=self.hparams.naf_enc_blk_nums, dec_blk_nums=self.hparams.naf_dec_blk_nums, dw_expand=self.hparams.naf_dw_expand, ffn_expand=self.hparams.naf_ffn_expand, block_type = NAFBlock3D)
        self.criterion = nn.MSELoss()  
        self.psnr = PeakSignalNoiseRatio(data_range=128)

    def forward(self, x, s=None):
        if s is not None:
            return self.model(x,s)
        return self.model(x)

    def on_fit_start(self):
        if self.hparams.ema == True:
            self.ema = ExponentialMovingAverage(self.model.parameters(), self.hparams.ema_rate)

    def training_step(self, batch, batch_idx):
        # Reverse batch if needed
        if self.hparams.style_size is not None:
            x, y, s = batch if not self.hparams.reversed else (batch[1], batch[0], batch)
        else:
            x, y = batch if not self.hparams.reversed else (batch[1], batch[0])

        if self.hparams.compressed:
            x = self.range_compression(x, div_factor = self.hparams.compression_factor, function = self.hparams.compression_type)
            y = self.range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_type)   

        if self.hparams.style_size is not None:
            y_hat = self(x,s)
        else:
            y_hat = self(x)

        if self.hparams.compressed:
            y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_type)
            y_hat = self.reverse_range_compression(y_hat, div_factor = self.hparams.compression_factor, function = self.hparams.compression_type)

        # Base lagrangian loss
        lag_loss = self.criterion(y_hat, y)
        train_loss = lag_loss

        # Flags for loss conditions
        eul_enabled = self.hparams.eul_loss

        # Euler loss component
        if eul_enabled:
            train_loss = torch.log(lag_loss) * self.hparams.lag_loss_scale
            eul_y_hat, eul_y = lag2eul([y_hat, y])
            eul_loss = self.criterion(eul_y_hat, eul_y)
            train_loss += torch.log(eul_loss) * self.hparams.eul_loss_scale


        # Logging
        train_psnr = self.psnr(y_hat, y)
        self.log('train_batch_psnr', train_psnr, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.log('train_epoch_psnr', train_psnr, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.log('train_batch_loss', train_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.log('train_epoch_loss', train_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_epoch_lag_loss', lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if eul_enabled:
            self.log('train_epoch_eul_loss', eul_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.style_size is not None:
            x, y, s = batch if not self.hparams.reversed else (batch[1], batch[0], batch)
        else:
            x, y = batch if not self.hparams.reversed else (batch[1], batch[0])

        if self.hparams.compressed:
            x = self.range_compression(x, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
            y = self.range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)   

        if self.hparams.style_size is not None:
            y_hat = self(x,s)
        else:
            y_hat = self(x)

        if self.hparams.compressed:
            y = self.reverse_range_compression(y, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)
            y_hat = self.reverse_range_compression(y_hat, div_factor = self.hparams.compression_factor, function = self.hparams.compression_factor)

        # Base lagrangian loss
        lag_loss = self.criterion(y_hat, y)
        val_loss = lag_loss

        # Flags for loss conditions
        eul_enabled = self.hparams.eul_loss

        # Euler loss component
        if eul_enabled:
            val_loss = torch.log(lag_loss) * self.hparams.lag_loss_scale
            eul_y_hat, eul_y = lag2eul([y_hat, y])
            eul_loss = self.criterion(eul_y_hat, eul_y)
            val_loss += torch.log(eul_loss) * self.hparams.eul_loss_scale


        # Logging
        val_psnr = self.psnr(y_hat, y)
        self.log('val_batch_psnr', val_psnr, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.log('val_epoch_psnr', val_psnr, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.log('val_batch_loss', val_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.log('val_epoch_loss', val_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_epoch_lag_loss', lag_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if eul_enabled:
            self.log('val_epoch_eul_loss', eul_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return y_hat

    def on_before_zero_grad(self, *args, **kwargs):
        # Update EMA after each training step, post-optimization
        if self.hparams.ema == True:
            self.ema.update()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer == 'Adamax':
            optimizer = optim.Adamax(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(), betas=(self.hparams.beta1, self.hparams.beta2), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_scheduler == 'Constant':
            return optimizer

        elif self.hparams.lr_scheduler == 'Cosine':
            if self.hparams.lr_cosine_period == None:
                total_steps = self.hparams.max_epochs * (self.hparams.num_samples // self.hparams.batch_size)
            else:
                total_steps = self.hparams.lr_cosine_period
            T_max = total_steps - self.hparams.lr_warmup
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.hparams.lr,
                    total_iters=self.hparams.lr_warmup,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max
                ),
            ],
            milestones=[self.hparams.lr_warmup],
        )
            scheduler = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": True,
            }
            return [optimizer], [scheduler]

        return optimizer

    def range_compression(self, sample, div_factor: float = 24, function: str = 'arcsinh', epsilon: float = 1e-8):
        """Applies compression on the input."""
        if function == 'arcsinh':
            return torch.arcsinh(sample / div_factor)*div_factor
        if function == 'tanh':
            return torch.tanh(sample / div_factor)*div_factor
        if function == 'sqrt':
            return torch.sign(sample)*torch.sqrt(torch.abs(sample + epsilon) / div_factor)*div_factor
        else:
            return sample  

    def reverse_range_compression(self, sample, div_factor: float = 24, function: str = 'arcsinh', epsilon: float = 1e-8):
        """Undos compression on the output."""
        if function == 'arcsinh':
            return torch.sinh(sample / div_factor)*div_factor
        if function == 'tanh':
            return torch.arctanh(torch.clamp((sample / div_factor),min=-0.999,max=0.999))*div_factor
        if function == 'sqrt':
            return torch.sign(sample)*torch.square((sample - epsilon) / div_factor)*div_factor  
        else:
            return sample
