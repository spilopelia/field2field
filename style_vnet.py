import torch
import torch.nn as nn

from style_block import ConvStyledBlock, ResStyledBlock

def narrow_by(a, c):
    """Narrow a by size c symmetrically on all edges.
    """
    ind = (slice(None),) * 2 + (slice(c, -c),) * (a.dim() - 2)
    return a[ind]

class StyledVNet(nn.Module):
    def __init__(self, style_size, in_chan, out_chan, bypass=None, **kwargs):
        """V-Net like network with styles

        See `vnet.VNet`.
        """
        super().__init__()

        # activate non-identity skip connection in residual block
        # by explicitly setting out_chan
        self.conv_l0 = ResStyledBlock(style_size, in_chan, 64, seq='CACBA')
        self.down_l0 = ConvStyledBlock(style_size, 64, seq='DBA')
        self.conv_l1 = ResStyledBlock(style_size, 64, 64, seq='CBACBA')
        self.down_l1 = ConvStyledBlock(style_size, 64, seq='DBA')

        self.conv_c = ResStyledBlock(style_size, 64, 64, seq='CBACBA')

        self.up_r1 = ConvStyledBlock(style_size, 64, seq='UBA')
        self.conv_r1 = ResStyledBlock(style_size, 128, 64, seq='CBACBA')
        self.up_r0 = ConvStyledBlock(style_size, 64, seq='UBA')
        self.conv_r0 = ResStyledBlock(style_size, 128, out_chan, seq='CAC')

        if bypass is None:
            self.bypass = in_chan == out_chan
        else:
            self.bypass = bypass

    def forward(self, x, s):
        if self.bypass:
            x0 = x

        y0 = self.conv_l0(x, s)
        x = self.down_l0(y0, s)

        y1 = self.conv_l1(x, s)
        x = self.down_l1(y1, s)

        x = self.conv_c(x, s)

        x = self.up_r1(x, s)
        y1 = narrow_by(y1, 4)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x, s)

        x = self.up_r0(x, s)
        y0 = narrow_by(y0, 16)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r0(x, s)

        if self.bypass:
            x0 = narrow_by(x0, 20)
            x += x0

        return x
    
class GeneralizedStyledVNet(nn.Module):
    def __init__(self, style_size, in_chan, out_chan, 
                 num_levels=2, base_channels=64, bypass=None,
                 r_half_reduction_per_resblock=2, **kwargs):
        super().__init__()
        assert num_levels >= 1, "num_levels must be at least 1."

        self.num_levels = num_levels
        self.base_channels = base_channels
        self.r_h = r_half_reduction_per_resblock 

        # --- Encoder Path ---
        self.encoder_conv_blocks = nn.ModuleList()
        self.encoder_downsample_blocks = nn.ModuleList()

        self.encoder_conv_blocks.append(
            ResStyledBlock(style_size, in_chan, base_channels, seq='CACBA')
        )
        current_c_enc = base_channels

        for i in range(num_levels): 
            self.encoder_downsample_blocks.append(
                ConvStyledBlock(style_size, current_c_enc, current_c_enc, seq='DBA')
            )
            if i < num_levels - 1: 
                self.encoder_conv_blocks.append(
                    ResStyledBlock(style_size, current_c_enc, current_c_enc, seq='CBACBA')
                )
        
        self.conv_c = ResStyledBlock(style_size, current_c_enc, current_c_enc, seq='CBACBA')

        # --- Decoder Path ---
        self.decoder_upsample_blocks = nn.ModuleList()
        self.decoder_conv_blocks = nn.ModuleList()
        
        current_c_dec = current_c_enc
        for i in range(num_levels - 1): 
            self.decoder_upsample_blocks.append(
                ConvStyledBlock(style_size, current_c_dec, current_c_dec, seq='UBA')
            )
            self.decoder_conv_blocks.append(
                ResStyledBlock(style_size, current_c_dec * 2, current_c_dec, seq='CBACBA')
            )
        
        self.up_r0 = ConvStyledBlock(style_size, current_c_dec, current_c_dec, seq='UBA')
        self.conv_r0 = ResStyledBlock(style_size, current_c_dec * 2, out_chan, seq='CAC')

        if bypass is None: self.bypass = (in_chan == out_chan)
        else: self.bypass = bypass
            
        # --- Calculate Crop Amounts ---
        N = self.num_levels
        self.skip_crop_amounts = []
        for k_level in range(N): # For y_k (k_level = 0 to N-1)
            # Formula: r_h * (3 * 2^(N-k) - 4). Smallest N-k is 1 (for k=N-1).
            # Smallest crop value is r_h * (3*2 - 4) = 2*r_h, which is non-negative.
            crop_val = self.r_h * (3 * (2**(N - k_level)) - 4)
            self.skip_crop_amounts.append(crop_val)
        
        if self.bypass:
            # Formula: r_h * (3 * 2^N - 2). Smallest N is 1.
            # Smallest crop value is r_h * (3*2 - 2) = 4*r_h, non-negative.
            self.bypass_crop_amount = self.r_h * (3 * (2**N) - 2)
        else:
            self.bypass_crop_amount = 0

    def forward(self, x, s):
        if self.bypass: x0_identity = x

        skip_connections = []
        x_path = x

        # --- Encoder ---
        for i in range(self.num_levels):
            y = self.encoder_conv_blocks[i](x_path, s)
            skip_connections.append(y)
            x_path = self.encoder_downsample_blocks[i](y, s) 
        
        x_processed = self.conv_c(x_path, s)

        # --- Decoder ---
        for i in range(self.num_levels - 1): 
            x_processed = self.decoder_upsample_blocks[i](x_processed, s)
            
            skip_idx = self.num_levels - 1 - i 
            skip = skip_connections[skip_idx]
            crop_amount = self.skip_crop_amounts[skip_idx]
            
            skip = narrow_by(skip, crop_amount)
            x_processed = torch.cat([skip, x_processed], dim=1)
            del skip 
            x_processed = self.decoder_conv_blocks[i](x_processed, s)

        x_processed = self.up_r0(x_processed, s)
        
        skip_l0 = skip_connections[0] 
        crop_amount_l0 = self.skip_crop_amounts[0]
        skip_l0 = narrow_by(skip_l0, crop_amount_l0)
        
        x_processed = torch.cat([skip_l0, x_processed], dim=1)
        del skip_l0; del skip_connections[:] 
        
        x_processed = self.conv_r0(x_processed, s)

        if self.bypass:
            x0_identity_narrowed = narrow_by(x0_identity, self.bypass_crop_amount)
            
            # Check for spatial mismatch before addition, especially if bypass path became zero-sized
            if any(x_processed.shape[d] != x0_identity_narrowed.shape[d] for d in range(x_processed.dim()) if d != 1):
                has_zero_dim_bypass = any(x0_identity_narrowed.shape[d] == 0 for d in range(2, x0_identity_narrowed.dim()))
                has_zero_dim_processed = any(x_processed.shape[d] == 0 for d in range(2, x_processed.dim()))

                if has_zero_dim_bypass and not has_zero_dim_processed:
                     raise RuntimeError(
                        f"Bypass path (x0_identity) has been narrowed to zero size in at least one spatial dimension, "
                        f"while main path (x_processed) is non-zero. "
                        f"x0_identity shape before narrow: {x0_identity.shape}, "
                        f"bypass_crop_amount: {self.bypass_crop_amount}, "
                        f"x0_identity_narrowed shape: {x0_identity_narrowed.shape}. "
                        f"x_processed shape: {x_processed.shape}. "
                        f"This usually means the input tensor is too small for the network's depth/cropping, "
                        f"or r_half_reduction_per_resblock ({self.r_h}) is misconfigured."
                    )
            
            x_processed += x0_identity_narrowed

        return x_processed