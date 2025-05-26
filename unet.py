import torch.nn as nn
import torch
from periodic_padding import periodic_padding_3d
from style_conv import ConvStyled3d_unet, FiLMLayer3D

def crop_tensor(x):
	x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
	return x

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

class StyledSequential(nn.Sequential):
    def forward(self, x, s):
        for module in self:
            x = module(x, s)
        return x
    
# Assuming conv3x3 and BasicBlock are defined as in your original code.
class BasicBlock(nn.Module):
	def __init__(self,inplane,outplane,stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
		self.bn1 = nn.BatchNorm3d(outplane)
		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		return out

class StyledBlock(nn.Module):
    def __init__(self, style_size, inplane, outplane, kernel_size=3, stride=1, resample=None):
        super().__init__()
        # Use ConvStyled3d_unet with style conditioning
        self.conv = ConvStyled3d_unet(style_size, inplane, outplane, kernel_size=kernel_size, stride=stride, resample=resample, bias=True)
        self.bn = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, s):
        # periodic padding same as before
        x = periodic_padding_3d(x, pad=(1,1,1,1,1,1))
        out = self.conv(x, s)  # pass style vector s here
        out = self.bn(out)
        out = self.relu(out)
        return out

class FiLMBlock(nn.Module):
    def __init__(self, style_size, inplane, outplane, kernel_size=3, stride=1, resample=None, bias=True):
        super().__init__()
        self.resample = resample

        if resample is None:
            self.conv = nn.Conv3d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
        elif resample == 'U': # Upsample
            # Stride here is the upsampling factor, typically 2
            self.conv = nn.ConvTranspose3d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
        elif resample == 'D': # Downsample
            # Stride here is the downsampling factor, typically 2
            self.conv = nn.Conv3d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
        else:
            raise ValueError(f"Unknown resample type: {resample}")

        self.bn = nn.BatchNorm3d(outplane)
        self.film = FiLMLayer3D(style_size, outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, s):
        # Consistent with your StyledBlock's padding
        x = periodic_padding_3d(x, pad=(1,1,1,1,1,1)) 
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.film(x, s)
        x = self.relu(x)
        return x
    
class BasicBlockwithnopadding(nn.Module):
	def __init__(self,inplane,outplane,stride = 1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
		self.bn1 = nn.BatchNorm3d(outplane)
		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		return out
     
class ResBlock(nn.Module):
    def __init__(self,inplane,outplane,stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        orig_x = x
        x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out + orig_x

class ResBlockwithnopadding(nn.Module):
    def __init__(self,inplane,outplane,stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        orig_x = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out + orig_x
    
class UNet3D(nn.Module):  
    def __init__(self, block=BasicBlock, num_layers=2, base_filters=64, blocks_per_layer=2,init_dim=3):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Decoder path
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.ConvTranspose3d(out_channels, init_dim, 1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        encoder_outputs = []

        x = self.init_conv(x)
        encoder_outputs.append(x)
        
        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)  # Compression layer
            x = self.encoders[i + 1](x)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))  # Assuming this is a custom function
            x = self.decoders[i](x)  # Transpose Conv layer
            x = crop_tensor(x)  # Assuming this is a custom function to crop the tensor           
            # Skip connection with encoder outputs
            x = torch.cat((x, encoder_outputs[len(encoder_outputs)-2-i//2]), dim=1)  # Skip connection
            x = self.decoders[i + 1](x)  # Non-compression layer

        # Final 1x1 Conv
        x = self.final_conv(x)
        return x

class StyledUNet3D(nn.Module):
    def __init__(self, style_size, num_layers=2, base_filters=64, blocks_per_layer=2, init_dim=3):
        super(StyledUNet3D, self).__init__()
        self.style_size = style_size
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        init_channels = init_dim
        out_channels = base_filters
        # Initial conv no resampling, stride=1
        self.init_conv = self._make_layer(style_size, init_channels, out_channels, blocks=blocks_per_layer, stride=1, resample=None)

        for _ in range(num_layers):
            # Downsample with resample='D' (stride=2)
            self.encoders.append(self._make_layer(style_size, out_channels, out_channels*2, blocks=1, stride=2, resample='D'))
            self.encoders.append(self._make_layer(style_size, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1, resample=None))
            out_channels *= 2

        for _ in range(num_layers):
            # Upsample with resample='U' (ConvTranspose)
            self.decoders.append(ConvStyled3d_unet(style_size, out_channels, out_channels//2, kernel_size=3, stride=2, resample='U', bias=True))
            self.decoders.append(self._make_layer(style_size, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1, resample=None))
            out_channels //= 2

        # Final conv (1x1 conv with no modulation, normal conv transpose)
        self.final_conv = ConvStyled3d_unet(style_size, out_channels, init_dim, kernel_size=1, stride=1, resample='U', bias=True)

    def _make_layer(self, style_size, inplanes, outplanes, blocks, stride, kernel_size=3, resample=None):
        layers = []
        for _ in range(blocks):
            layers.append(StyledBlock(style_size, inplanes, outplanes, kernel_size, stride, resample))
            inplanes = outplanes
            # After first block, resample only in first block of layer
            resample = None
        return StyledSequential(*layers)
    
    def forward(self, x, s):
        encoder_outputs = []

        x = self.init_conv(x, s)
        encoder_outputs.append(x)

        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x, s)  # Compression (downsample) layer with modulation
            x = self.encoders[i+1](x, s)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))
            x = self.decoders[i](x, s)  # Upsample layer with modulation
            x = crop_tensor(x)
            skip = encoder_outputs[len(encoder_outputs) - 2 - i // 2]
            x = torch.cat((x, skip), dim=1)
            x = self.decoders[i + 1](x, s)

        x = self.final_conv(x, s)
        return x

class FiLMUNet3D(nn.Module):
    def __init__(self, style_size, num_layers=2, base_filters=64, blocks_per_layer=2, init_dim=3):
        super(FiLMUNet3D, self).__init__()
        self.style_size = style_size
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        init_channels = init_dim
        out_channels = base_filters
        # Initial conv no resampling, stride=1
        self.init_conv = self._make_layer(style_size, init_channels, out_channels, blocks=blocks_per_layer, stride=1, resample=None)

        for _ in range(num_layers):
            # Downsample with resample='D' (stride=2)
            self.encoders.append(self._make_layer(style_size, out_channels, out_channels*2, blocks=1, stride=2, resample='D'))
            self.encoders.append(self._make_layer(style_size, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1, resample=None))
            out_channels *= 2
        for _ in range(num_layers):
            # Kernel_size=3, stride=2, padding=0 for ConvTranspose to match original UNet3D upsampler behavior
            # and direct ConvStyled3d_unet(resample='U') behavior
            upsampler_conv = nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=3, stride=2, padding=0, bias=True)
            upsampler_film = FiLMLayer3D(style_size, out_channels//2)
            self.decoders.append(nn.ModuleList([upsampler_conv, upsampler_film]))
            self.decoders.append(self._make_layer(style_size, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1, resample=None))
            out_channels //= 2

        # Final conv (1x1 conv with no modulation, normal conv transpose)
        final_conv_layer  = nn.ConvTranspose3d(out_channels, init_dim, 1, stride=1, padding=0)
        final_film_layer = FiLMLayer3D(style_size, init_dim)
        self.final_conv = nn.ModuleList([final_conv_layer, final_film_layer])

    def _make_layer(self, style_size, inplanes, outplanes, blocks, stride, kernel_size=3, resample=None):
        layers = []
        for _ in range(blocks):
            layers.append(FiLMBlock(style_size, inplanes, outplanes, kernel_size, stride, resample))
            inplanes = outplanes
            # After first block, resample only in first block of layer
            resample = None
        return StyledSequential(*layers)
    
    def forward(self, x, s):
        encoder_outputs = []

        x = self.init_conv(x, s)
        encoder_outputs.append(x)

        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x, s)  # Compression (downsample) layer with modulation
            x = self.encoders[i+1](x, s)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))
            upsampler_conv, upsampler_film = self.decoders[i]
            x = upsampler_conv(x)
            x = upsampler_film(x, s)
            x = crop_tensor(x)
            skip = encoder_outputs[len(encoder_outputs) - 2 - i // 2]
            x = torch.cat((x, skip), dim=1)
            conv_blocks = self.decoders[i + 1]
            x = conv_blocks(x, s)

        final_conv_layer, final_film_layer = self.final_conv
        x = final_conv_layer(x)
        x = final_film_layer(x, s)
        return x
    
class UNet3Dwithnopadding(nn.Module):  
    def __init__(self, block=BasicBlockwithnopadding, num_layers=2, base_filters=64, blocks_per_layer=2,init_dim=3):
        super(UNet3Dwithnopadding, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Decoder path
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.ConvTranspose3d(out_channels, init_dim, 1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        encoder_outputs = []

        x = self.init_conv(x)
        encoder_outputs.append(x)
        
        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)  # Compression layer
            x = self.encoders[i + 1](x)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)  # Transpose Conv layer
            x = torch.cat((x, encoder_outputs[len(encoder_outputs)-2-i//2]), dim=1)  # Skip connection
            x = self.decoders[i + 1](x)  # Non-compression layer

        # Final 1x1 Conv
        x = self.final_conv(x)
        return x
    
class UNet3DwithRes(nn.Module):  
    def __init__(self, block=BasicBlock, num_layers=2, base_filters=64, blocks_per_layer=2,init_dim=3):
        super(UNet3DwithRes, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Decoder path
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.ConvTranspose3d(out_channels, init_dim, 1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            if inplanes == outplanes:
                layers.append(ResBlock(inplanes, outplanes, stride=stride))
            else:
                layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        encoder_outputs = []

        x = self.init_conv(x)
        encoder_outputs.append(x)
        
        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)  # Compression layer
            x = self.encoders[i + 1](x)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = periodic_padding_3d(x, pad=(0, 1, 0, 1, 0, 1))  # Assuming this is a custom function
            x = self.decoders[i](x)  # Transpose Conv layer
            x = crop_tensor(x)  # Assuming this is a custom function to crop the tensor
            
            x = torch.cat((x, encoder_outputs[len(encoder_outputs)-2-i//2]), dim=1)  # Skip connection
            
            x = self.decoders[i + 1](x)  # Non-compression layer

        # Final 1x1 Conv
        x = self.final_conv(x)
        return x

class UNet3DwithResnopadding(nn.Module):  
    def __init__(self, block=ResBlockwithnopadding, num_layers=2, base_filters=64, blocks_per_layer=2,init_dim=3):
        super(UNet3DwithResnopadding, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder path
        init_channels = init_dim
        out_channels = base_filters
        self.init_conv = self._make_layer(block, init_channels, out_channels, blocks=blocks_per_layer, stride=1)
        for _ in range(num_layers):
            self.encoders.append(self._make_layer(block, out_channels, out_channels*2, blocks=1, stride=2))
            self.encoders.append(self._make_layer(block, out_channels*2, out_channels*2, blocks=blocks_per_layer, stride=1))
            out_channels *= 2

        # Decoder path
        for _ in range(num_layers):
            self.decoders.append(nn.ConvTranspose3d(out_channels, out_channels//2, kernel_size=3, stride=2, padding=0))
            self.decoders.append(self._make_layer(block, out_channels, out_channels//2, blocks=blocks_per_layer, stride=1))
            out_channels //= 2

        self.final_conv = nn.ConvTranspose3d(out_channels, init_dim, 1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=1):
        layers = []
        for _ in range(blocks):
            if inplanes == outplanes:
                layers.append(ResBlock(inplanes, outplanes, stride=stride))
            else:
                layers.append(block(inplanes, outplanes, stride=stride))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        encoder_outputs = []

        x = self.init_conv(x)
        encoder_outputs.append(x)
        
        # Encoding path
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)  # Compression layer
            x = self.encoders[i + 1](x)  # Non-compression layer
            encoder_outputs.append(x)

        # Decoding path
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)  # Transpose Conv layer            
            x = torch.cat((x, encoder_outputs[len(encoder_outputs)-2-i//2]), dim=1)  # Skip connection
            x = self.decoders[i + 1](x)  # Non-compression layer

        # Final 1x1 Conv
        x = self.final_conv(x)
        return x

