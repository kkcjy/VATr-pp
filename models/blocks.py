import torch
import torch.nn.functional as F
from torch import nn


class ResBlocks(nn.Module):
    def __init__(self, n_blocks, dim, norm, act, pad_type):
        super(ResBlocks, self).__init__()
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(dim, norm=norm, activation=act, pad_type=pad_type))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        layers = []
        layers.append(ConvBlock(dim, dim, 3, 1, 1,
                                norm=norm,
                                activation=activation,
                                pad_type=pad_type))
        layers.append(ConvBlock(dim, dim, 3, 1, 1,
                                norm=norm,
                                activation='none',
                                pad_type=pad_type))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class ActFirstResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hid_ch=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.use_shortcut = (in_ch != out_ch)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hid_ch = min(in_ch, out_ch) if hid_ch is None else hid_ch
        
        self.conv0 = ConvBlock(self.in_ch, self.hid_ch, 3, 1,
                               padding=1, pad_type='reflect', norm=norm,
                               activation=activation, act_first=True)
        self.conv1 = ConvBlock(self.hid_ch, self.out_ch, 3, 1,
                               padding=1, pad_type='reflect', norm=norm,
                               activation=activation, act_first=True)
        if self.use_shortcut:
            self.conv_s = ConvBlock(self.in_ch, self.out_ch, 1, 1,
                                    activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.use_shortcut else x
        dx = self.conv0(x)
        dx = self.conv1(dx)
        return x_s + dx


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # init normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        # init activation
        if activation == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'none':
            self.act = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, act_first=False):
        super(ConvBlock, self).__init__()
        self.use_bias = use_bias
        self.act_first = act_first
        
        # init padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported pad type: {pad_type}")

        # init normalization
        norm_dim = out_ch
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        # init activation
        if activation == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'none':
            self.act = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, bias=self.use_bias)

    def forward(self, x):
        if self.act_first:
            if self.act:
                x = self.act(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.act:
                x = self.act(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, n_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.n_features = n_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(n_features))
        self.register_buffer('running_var', torch.ones(n_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.n_features) + ')'