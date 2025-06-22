import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Pixelwise normalization after conv layers.

    See ProGAN, StyleGAN.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


class LinearElr(nn.Module):
    """Linear layer with equalized learning rate.

    See ProGAN, StyleGAN, and 1706.05350

    Useful at all if not for regularization(1706.05350)?
    """
    def __init__(self, in_size, out_size, bias=True, act=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        self.wnorm = 1 / math.sqrt(in_size)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_size))
        else:
            self.register_parameter('bias', None)

        self.act = act

    def forward(self, x):
        x = F.linear(x, self.weight * self.wnorm, bias=self.bias)

        if self.act:
            x = F.leaky_relu(x, negative_slope=0.2)

        return x


class ConvElr3d(nn.Module):
    """Conv3d layer with equalized learning rate.

    See ProGAN, StyleGAN, and 1706.05350

    Useful at all if not for regularization(1706.05350)?
    """
    def __init__(self, in_chan, out_chan, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_chan, in_chan, *(kernel_size,) * 3),
        )
        fan_in = in_chan * kernel_size ** 3
        self.wnorm = 1 / math.sqrt(fan_in)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
        else:
            self.register_parameter('bias', None)

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = F.conv2d(
            x,
            self.weight * self.wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return x


class ConvStyled3d(nn.Module):
    """Convolution layer with modulation and demodulation, from StyleGAN2.

    Weight and bias initialization from `torch.nn._ConvNd.reset_parameters()`.
    """
    def __init__(self, style_size, in_chan, out_chan, kernel_size=3, stride=1,
                 bias=True, resample=None):
        super().__init__()

        self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1

        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        elif resample == 'U':
            K3 = (2,) * 3
            # NOTE not clear to me why convtranspose have channels swapped
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
            self.stride = 2
            self.conv = F.conv_transpose3d
        elif resample == 'D':
            K3 = (2,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = 2
            self.conv = F.conv3d
        else:
            raise ValueError('resample type {} not supported'.format(resample))
        self.resample = resample

        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5),
            mode='fan_in',  # effectively 'fan_out' for 'D'
            nonlinearity='leaky_relu',
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, s, eps=1e-8):
        N, Cin, *DHWin = x.shape
        C0, C1, *K3 = self.weight.shape
        if self.resample == 'U':
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        s = F.linear(s, self.style_weight, bias=self.style_bias)

        # modulation
        if self.resample == 'U':
            s = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s = s.reshape(N, 1, Cin, 1, 1, 1)
        w = self.weight * s

        # demodulation
        if self.resample == 'U':
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        w = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        w = w.reshape(N * C0, C1, *K3)
        x = x.reshape(1, N * Cin, *DHWin)
        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=N)
        _, _, *DHWout = x.shape
        x = x.reshape(N, Cout, *DHWout)

        return x

class ConvStyled3d_unet(nn.Module):
    """Convolution layer with modulation and demodulation, from StyleGAN2.

    Weight and bias initialization from `torch.nn._ConvNd.reset_parameters()`.
    """
    def __init__(self, style_size, in_chan, out_chan, kernel_size=3, stride=1,
                 bias=True, resample=None):
        super().__init__()

        self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(in_chan))  # NOTE: init to 1

        if resample is None:
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        elif resample == 'U':
            K3 = (kernel_size,) * 3
            # NOTE not clear to me why convtranspose have channels swapped
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, *K3))
            self.stride = stride
            self.conv = F.conv_transpose3d
        elif resample == 'D':
            K3 = (kernel_size,) * 3
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, *K3))
            self.stride = stride
            self.conv = F.conv3d
        else:
            raise ValueError('resample type {} not supported'.format(resample))
        self.resample = resample

        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5),
            mode='fan_in',  # effectively 'fan_out' for 'D'
            nonlinearity='leaky_relu',
        )

        if bias:
            #self.bias = nn.Parameter(torch.zeros(out_chan))
            #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            #bound = 1 / math.sqrt(fan_in)
            #nn.init.uniform_(self.bias, -bound, bound)
            self.bias = nn.Parameter(torch.zeros(1, out_chan, 1, 1, 1))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, s, eps=1e-8):
        N, Cin, *DHWin = x.shape
        C0, C1, *K3 = self.weight.shape
        if self.resample == 'U':
            Cin, Cout = C0, C1
        else:
            Cout, Cin = C0, C1

        s = F.linear(s, self.style_weight, bias=self.style_bias)

        # modulation
        if self.resample == 'U':
            s = s.reshape(N, Cin, 1, 1, 1, 1)
        else:
            s = s.reshape(N, 1, Cin, 1, 1, 1)
        w = self.weight * s

        # demodulation
        if self.resample == 'U':
            fan_in_dim = (1, 3, 4, 5)
        else:
            fan_in_dim = (2, 3, 4, 5)
        w = w * torch.rsqrt(w.pow(2).sum(dim=fan_in_dim, keepdim=True) + eps)

        w = w.reshape(N * C0, C1, *K3)
        x = x.reshape(1, N * Cin, *DHWin)
        #x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=N)
        x = self.conv(x, w, bias=None, stride=self.stride, groups=N)
        _, _, *DHWout = x.shape
        x = x.reshape(N, Cout, *DHWout)

        if self.bias is not None:
            x = x + self.bias # Broadcasting (1,Cout_p,1,1,1) with (N,Cout_p,D,H,W)

        return x

class BatchNormStyled3d(nn.BatchNorm3d) :
    """ Trivially does standard batch normalization, but accepts second argument

    for style array that is not used
    """
    def forward(self, x, s):
        return super().forward(x)

class LeakyReLUStyled(nn.LeakyReLU):
    """ Trivially evaluates standard leaky ReLU, but accepts second argument

    for sytle array that is not used
    """
    def forward(self, x, s):
        return super().forward(x)
    
class FiLMLayer3D(nn.Module):
    """
    Feature-wise Linear Modulation Layer for 3D.
    """
    def __init__(self, style_size, num_features):
        super().__init__()
        self.num_features = num_features
        self.style_projection = nn.Linear(style_size, num_features * 2)

    def forward(self, x, s):
        # x: (N, C, D, H, W)
        # s: (N, style_size)
        
        # Project style vector to get gamma and beta
        style_params = self.style_projection(s) # (N, C*2)
        gamma = style_params[:, :self.num_features] # (N, C)
        beta = style_params[:, self.num_features:]  # (N, C)

        # Reshape gamma and beta for broadcasting: (N, C, 1, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta
    
class ModulatedConv3d(nn.Module):
    """Modulated 3D Convolution layer, inspired by StyleGAN2.
    """
    def __init__(self, style_size, in_chan, out_chan, kernel_size=3, padding=0, stride=1,
                 bias=True, resample=None, eps=1e-8):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.eps = eps
        self.resample = resample # 'U' for upsample (ConvTranspose3d), 'D' for downsample (Conv3d with stride > 1)

        # Style modulation parameters
        self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(in_chan)) # Init to 1 for initial identity-like modulation

        # Convolution weights
        if self.resample == 'U': # ConvTranspose3d
            # NOTE: PyTorch ConvTranspose3d weights are (in_chan, out_chan/groups, *kernel_size)
            self.weight = nn.Parameter(torch.empty(in_chan, out_chan, kernel_size, kernel_size, kernel_size))
            self.conv_fn = F.conv_transpose3d
        else: # Conv3d (covers no resampling or 'D' for downsampling)
            self.weight = nn.Parameter(torch.empty(out_chan, in_chan, kernel_size, kernel_size, kernel_size))
            self.conv_fn = F.conv3d
        
        nn.init.kaiming_uniform_(
            self.weight, a=math.sqrt(5),
            mode='fan_in', # For ConvTranspose3d, fan_in is out_chan; for Conv3d, fan_in is in_chan * K^3
            nonlinearity='leaky_relu'
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_chan, 1, 1, 1))
            # Optional: Initialize bias similar to nn.Conv3d if desired
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # if fan_in > 0:
            #     bound = 1 / math.sqrt(fan_in)
            #     nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, s):
        N, Cin, *DHWin = x.shape

        # (N, style_size) -> (N, in_chan) for modulation
        style_modulation = F.linear(s, self.style_weight, bias=self.style_bias)

        if self.resample == 'U': # ConvTranspose3d
            # Modulate each input channel's filters
            w = self.weight.unsqueeze(0) * style_modulation.view(N, self.in_chan, 1, 1, 1, 1)
            # Demodulate: normalize over [out_chan_per_group, K, K, K] dimensions
            # w shape: (N, I, O, *K)
            demod_dims = (2, 3, 4, 5) # Demodulate over O, K, K, K
        else: # Conv3d
            # Modulate each input channel's contribution to output filters
            w = self.weight.unsqueeze(0) * style_modulation.view(N, 1, self.in_chan, 1, 1, 1)
            # Demodulate: normalize over [in_chan_per_group, K, K, K] dimensions
            # w shape: (N, O, I, *K)
            demod_dims = (2, 3, 4, 5) # Demodulate over I, K, K, K

        w = w * torch.rsqrt(w.pow(2).sum(dim=demod_dims, keepdim=True) + self.eps)

        x = x.view(1, N * Cin, *DHWin) 
        
        if self.resample == 'U': # ConvTranspose3d
            kernels = w.view(N * self.in_chan, self.out_chan, *self.weight.shape[2:])
        else: # Conv3d
            kernels = w.view(N * self.out_chan, self.in_chan, *self.weight.shape[2:])

        # Perform convolution
        if self.resample == 'U':
            # For conv_transpose3d, padding means something different.
            # Output padding might be needed if stride > 1 to get precise output shape.
            # For stride=1, kernel_size=K, padding=(K-1)/2 preserves size.
            out = self.conv_fn(x, kernels, bias=None, stride=self.stride, padding=self.padding, groups=N)
        else: # Conv3d
            out = self.conv_fn(x, kernels, bias=None, stride=self.stride, padding=self.padding, groups=N)

        # Reshape output
        # Output from conv is (1, N*Cout, D', H', W')
        _, _, *DHWout = out.shape
        out = out.view(N, self.out_chan, *DHWout)

        if self.bias is not None:
            out = out + self.bias # (N, Cout, D', H', W') + (1, Cout, 1, 1, 1)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_chan={self.in_chan}, out_chan={self.out_chan}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"resample={self.resample}, style_size={self.style_weight.shape[1]})"
        )
    
class DepthwiseModulatedConv3d(nn.Module):
    """Modulated Depthwise 3D Convolution layer."""
    def __init__(self, style_size, in_chan, out_chan, kernel_size=3, padding=0, stride=1, bias=True, eps=1e-8):
        super().__init__()
        if out_chan != in_chan:
            print(f"Warning: For depthwise=True, out_chan ({out_chan}) is ignored and set to in_chan ({self.in_chan}).")
        channels = in_chan
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.eps = eps

        # Style modulation parameters
        self.style_weight = nn.Parameter(torch.empty(channels, style_size))
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(channels))

        # Convolution weights (depthwise)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, s):
        N, C, D, H, W = x.shape

        # Style modulation: (N, style_size) -> (N, C)
        style_modulation = F.linear(s, self.style_weight, bias=self.style_bias)

        # Modulate weights: (N, C, 1, K, K, K)
        w = self.weight.unsqueeze(0) * style_modulation.view(N, C, 1, 1, 1, 1)

        # Demodulation
        demod = torch.rsqrt(w.pow(2).sum(dim=[2, 3, 4, 5], keepdim=True) + self.eps)
        w = w * demod

        # Reshape input and weights for group convolution
        x = x.view(1, N * C, D, H, W)
        w = w.view(N * C, 1, self.kernel_size, self.kernel_size, self.kernel_size)

        out = F.conv3d(x, w, bias=None, stride=self.stride, padding=self.padding, groups=N * C)

        # Reshape output
        _, _, D_out, H_out, W_out = out.shape
        out = out.view(N, C, D_out, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}(channels={self.channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, style_size={self.style_weight.shape[1]})")
