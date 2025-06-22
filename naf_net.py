import torch
import torch.nn as nn
import torch.nn.functional as F
from style_conv import ModulatedConv3d, DepthwiseModulatedConv3d

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, D, H, W)
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = self.weight * x + self.bias
        return x

class PixelShuffle3D(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        channels //= self.scale_factor ** 3
        x = x.view(batch_size, channels, self.scale_factor, self.scale_factor, self.scale_factor, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels, 
                   depth * self.scale_factor, 
                   height * self.scale_factor, 
                   width * self.scale_factor)
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class BaselineBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv3d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dw_channel, dw_channel // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dw_channel // 2, dw_channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv3d(ffn_channel, c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class BaselineBlock_SG(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv3d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=dw_channel // 2, out_channels=dw_channel // 4, kernel_size=1, padding=0, stride=1, # Bottleneck adjusted
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=dw_channel // 4, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, # Output matches SG output
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv3d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class BaselineBlock_SCA(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv3d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv3d(ffn_channel, c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class BaselineBlock_SCA_Modulated(nn.Module):
    def __init__(self, c, style_dim, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Replace nn.Conv3d with DepthwiseModulatedConv3d
        self.mod_conv2 = DepthwiseModulatedConv3d(
            style_size=style_dim, 
            in_chan=dw_channel, 
            out_chan=dw_channel, 
            kernel_size=3, 
            padding=1,  # To maintain spatial dimensions with kernel_size=3, stride=1
            stride=1, 
            bias=True
        )
        
        self.conv3 = nn.Conv3d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv3d(ffn_channel, c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm3d(c) # Expects (N, C, D, H, W), normalizes over C
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp, style_vector): # Added style_vector as input
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        # Use the modulated convolution
        x = self.mod_conv2(x, style_vector) # Pass the style vector here
        x = self.gelu(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class BaselineBlock_SCA_FullyModulated(nn.Module):
    def __init__(self, c, style_dim, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        # Convolution 1 (1x1)
        self.mod_conv1 = ModulatedConv3d(
            style_size=style_dim, 
            in_chan=c, 
            out_chan=dw_channel, 
            kernel_size=1, 
            padding=0, 
            stride=1, 
            bias=True
        )
        
        # Convolution 2 (3x3) - Note: This is a depthwise conv
        self.mod_conv2 = DepthwiseModulatedConv3d(
            style_size=style_dim, 
            in_chan=dw_channel, 
            out_chan=dw_channel, 
            kernel_size=3, 
            padding=1,
            stride=1, 
            bias=True
        )
        
        # Convolution 3 (1x1)
        self.mod_conv3 = ModulatedConv3d(
            style_size=style_dim, 
            in_chan=dw_channel, 
            out_chan=c, 
            kernel_size=1, 
            padding=0, 
            stride=1, 
            bias=True
        )
        
        # Simplified Channel Attention (SCA) parts
        self.sca_pool = nn.AdaptiveAvgPool3d(1)
        self.sca_conv = ModulatedConv3d(
            style_size=style_dim,
            in_chan=dw_channel, 
            out_chan=dw_channel, 
            kernel_size=1, 
            padding=0, 
            stride=1,
            bias=True
        )
        
        self.gelu = nn.GELU()

        # Feed Forward Network (FFN) parts
        ffn_channel = FFN_Expand * c
        self.mod_conv4 = ModulatedConv3d(
            style_size=style_dim,
            in_chan=c, 
            out_chan=ffn_channel, 
            kernel_size=1, 
            padding=0, # K=1, P=0, S=1
            stride=1,
            bias=True
        )
        self.mod_conv5 = ModulatedConv3d(
            style_size=style_dim,
            in_chan=ffn_channel, 
            out_chan=c, 
            kernel_size=1, 
            padding=0, # K=1, P=0, S=1
            stride=1,
            bias=True
        )

        self.norm1 = LayerNorm3d(c) # Normalizes over C channels
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp, style_vector): # Added style_vector as input
        x_shortcut = inp # Store for the first residual connection

        # First part of the block
        x = self.norm1(inp)
        
        x = self.mod_conv1(x, style_vector)
        x = self.mod_conv2(x, style_vector) # This is now a full modulated convolution
        x = self.gelu(x)
        
        # Apply SCA
        sca_attention = self.sca_pool(x)
        sca_attention = self.sca_conv(sca_attention, style_vector)
        x = x * sca_attention
        
        x = self.mod_conv3(x, style_vector)
        x = self.dropout1(x)
        
        # First residual connection
        y = x_shortcut + x * self.beta

        # Second part of the block (FFN)
        x_ffn_shortcut = y # Store for the second residual connection
        
        x = self.norm2(y)
        x = self.mod_conv4(x, style_vector)
        x = self.gelu(x)
        x = self.mod_conv5(x, style_vector)
        x = self.dropout2(x)

        # Second residual connection
        out = x_ffn_shortcut + x * self.gamma
        return out
        
class NAFBlock3D(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv3d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv3d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma

class NAFNet3D(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, 
                 enc_blk_nums=[], dec_blk_nums=[], dw_expand=2, ffn_expand=2, block_type = BaselineBlock):
        super().__init__()
        self.intro = nn.Conv3d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv3d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width    
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[block_type(chan, dw_expand, ffn_expand) for _ in range(num)]))
            self.downs.append(nn.Conv3d(chan, 2*chan, kernel_size=2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[block_type(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv3d(chan, chan * 4, kernel_size=1, bias=False),
                PixelShuffle3D(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[block_type(chan, dw_expand, ffn_expand) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, D, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :D, :H, :W]

    def check_image_size(self, x):
        _, _, d, h, w = x.size()
        mod_pad_d = (self.padder_size - d % self.padder_size) % self.padder_size
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d))
    
class NAFNet3D_modulated(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=4, 
                 enc_blk_nums=None, dec_blk_nums=None, 
                 dw_expand=2, ffn_expand=2, drop_out_rate=0.,
                 block_type=BaselineBlock_SCA, 
                 style_dim=None,
                 modulate_outer_convs=False): # New flag
        super().__init__()

        if enc_blk_nums is None: enc_blk_nums = [2, 2, 4, 8] # Example default from NAFNet paper
        if dec_blk_nums is None: dec_blk_nums = [2, 2, 2, 2] # Example default

        self.img_channel = img_channel
        self.width = width
        self.dw_expand = dw_expand
        self.ffn_expand = ffn_expand
        self.drop_out_rate = drop_out_rate
        self.block_type_is_modulated = (block_type == BaselineBlock_SCA_FullyModulated or block_type == BaselineBlock_SCA_Modulated)
        self.modulate_outer_convs = modulate_outer_convs
        self.style_dim = style_dim

        # Determine if any part of the network uses style modulation
        self.overall_uses_modulation = self.block_type_is_modulated or self.modulate_outer_convs

        if self.overall_uses_modulation and self.style_dim is None:
            raise ValueError("`style_dim` must be provided if `block_type` is modulated or `modulate_outer_convs` is True.")

        # --- Intro Layer ---
        if self.modulate_outer_convs:
            self.intro = ModulatedConv3d(style_size=self.style_dim, in_chan=img_channel, out_chan=width, 
                                         kernel_size=3, padding=1, stride=1, bias=True)
        else:
            self.intro = nn.Conv3d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, 
                                   groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList() # Kept as ModuleList for consistency if needed later
        
        self.downs = nn.ModuleList()
        self.ups_convs = nn.ModuleList()
        self.pixel_shuffles = nn.ModuleList() # PixelShuffle is not modulated

        chan = width    
        # --- Encoder Path ---
        for num in enc_blk_nums:
            encoder_blocks = []
            for _ in range(num):
                if self.block_type_is_modulated:
                    encoder_blocks.append(block_type(c=chan, style_dim=self.style_dim, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate))
                else:
                    encoder_blocks.append(block_type(c=chan, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate))
            self.encoders.append(nn.Sequential(*encoder_blocks))
            
            # Downsampling Layer
            if self.modulate_outer_convs:
                self.downs.append(ModulatedConv3d(style_size=self.style_dim, in_chan=chan, out_chan=2*chan, 
                                                  kernel_size=2, stride=2, padding=0, bias=True))
            else:
                self.downs.append(nn.Conv3d(chan, 2*chan, kernel_size=2, stride=2, padding=0))
            chan *= 2
        
        # --- Middle Blocks ---
        middle_blocks_seq = []
        for _ in range(middle_blk_num):
            if self.block_type_is_modulated:
                middle_blocks_seq.append(block_type(c=chan, style_dim=self.style_dim, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate))
            else:
                middle_blocks_seq.append(block_type(c=chan, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate))
        self.middle_blks_seq = nn.Sequential(*middle_blocks_seq) # Use nn.Sequential here

        # --- Decoder Path ---
        for num in dec_blk_nums:
            # Upsampling Layer (Conv part)
            if self.modulate_outer_convs:
                self.ups_convs.append(ModulatedConv3d(style_size=self.style_dim, in_chan=chan, out_chan=chan * 4, # For PixelShuffle3D factor 2
                                                      kernel_size=1, padding=0, bias=False)) # Bias often False before norm/shuffle
            else:
                self.ups_convs.append(nn.Conv3d(chan, chan * 4, kernel_size=1, padding=0, bias=False))
            self.pixel_shuffles.append(PixelShuffle3D(2))
            
            chan //= 2 # Channel reduction after upsampling and before decoder blocks for that level
            
            decoder_blocks = []
            for _ in range(num):
                if self.block_type_is_modulated:
                    decoder_blocks.append(block_type(c=chan, style_dim=self.style_dim, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate))
                else:
                    decoder_blocks.append(block_type(c=chan, DW_Expand=dw_expand, FFN_Expand=ffn_expand, drop_out_rate=drop_out_rate))
            self.decoders.append(nn.Sequential(*decoder_blocks))

        # --- Ending Layer ---
        if self.modulate_outer_convs:
            self.ending = ModulatedConv3d(style_size=self.style_dim, in_chan=width, out_chan=img_channel, 
                                          kernel_size=3, padding=1, stride=1, bias=True)
        else:
            self.ending = nn.Conv3d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, 
                                    groups=1, bias=True)

        self.padder_size = 2 ** len(self.encoders)

    def _apply_block_sequence(self, module_sequence, x, style_vector):
        if self.block_type_is_modulated: # This refers to the type of blocks in the sequence
            if style_vector is None:
                raise ValueError("`style_vector` must be provided for modulated blocks during forward pass.")
            for block in module_sequence: # module_sequence is nn.Sequential
                x = block(x, style_vector)
        else:
            x = module_sequence(x)
        return x

    def forward(self, inp, style_vector=None):
        if self.overall_uses_modulation and style_vector is None:
            raise ValueError("`style_vector` must be provided if any part of the network is modulated.")

        B, C, D_orig, H_orig, W_orig = inp.shape
        inp_padded = self.check_image_size(inp)

        # --- Intro ---
        if self.modulate_outer_convs:
            x = self.intro(inp_padded, style_vector)
        else:
            x = self.intro(inp_padded)
        
        encs = []

        # --- Encoder ---
        for encoder_seq, down_layer in zip(self.encoders, self.downs):
            x = self._apply_block_sequence(encoder_seq, x, style_vector)
            encs.append(x)
            if self.modulate_outer_convs:
                x = down_layer(x, style_vector)
            else:
                x = down_layer(x)

        # --- Middle ---
        x = self._apply_block_sequence(self.middle_blks_seq, x, style_vector)

        # --- Decoder ---
        for decoder_seq, up_conv_layer, ps_layer, enc_skip in zip(self.decoders, self.ups_convs, self.pixel_shuffles, encs[::-1]):
            if self.modulate_outer_convs:
                x = up_conv_layer(x, style_vector)
            else:
                x = up_conv_layer(x)
            x = ps_layer(x)
            
            x = x + enc_skip # Skip connection
            x = self._apply_block_sequence(decoder_seq, x, style_vector)

        # --- Ending ---
        if self.modulate_outer_convs:
            x = self.ending(x, style_vector)
        else:
            x = self.ending(x)
            
        x = x + inp_padded # Final skip connection

        return x[:, :, :D_orig, :H_orig, :W_orig] # Crop to original size

    def check_image_size(self, x):
        _, _, d, h, w = x.size()
        mod_pad_d = (self.padder_size - d % self.padder_size) % self.padder_size
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_d == 0 and mod_pad_h == 0 and mod_pad_w == 0:
            return x
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d))