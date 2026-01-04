import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 out_blocks=[DEFAULT_BLOCK_INDEX],
                 resize=True,
                 normalize=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        out_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model.
        normalize : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation.
        """
        super(InceptionV3, self).__init__()

        self.resize = resize
        self.normalize = normalize
        self.out_blocks = sorted(out_blocks)
        self.last_block = max(out_blocks)

        assert self.last_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """Get Inception feature maps

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.Tensor, corresponding to the selected output
        block, sorted ascending by index
        """
        out = []
        
        if self.resize:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_blocks:
                out.append(x)

            if idx == self.last_block:
                break

        return out


def fid_inception_v3():
    """Build pretrained Inception model for FID computation"""
    model = models.inception_v3(num_classes=1008,
                                aux_logits=False,
                                pretrained=False)
    model.Mixed_5b = FIDInceptionA(192, pool_features=32)
    model.Mixed_5c = FIDInceptionA(256, pool_features=64)
    model.Mixed_5d = FIDInceptionA(288, pool_features=64)
    model.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    model.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    model.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    model.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    model.Mixed_7b = FIDInceptionE_1(1280)
    model.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    model.load_state_dict(state_dict)
    return model


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        b1 = self.branch1x1(x)

        b5 = self.branch5x5_1(x)
        b5 = self.branch5x5_2(b5)

        b3 = self.branch3x3dbl_1(x)
        b3 = self.branch3x3dbl_2(b3)
        b3 = self.branch3x3dbl_3(b3)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                            count_include_pad=False)
        pool = self.branch_pool(pool)

        return torch.cat([b1, b5, b3, pool], 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        b1 = self.branch1x1(x)

        b7 = self.branch7x7_1(x)
        b7 = self.branch7x7_2(b7)
        b7 = self.branch7x7_3(b7)

        b7dbl = self.branch7x7dbl_1(x)
        b7dbl = self.branch7x7dbl_2(b7dbl)
        b7dbl = self.branch7x7dbl_3(b7dbl)
        b7dbl = self.branch7x7dbl_4(b7dbl)
        b7dbl = self.branch7x7dbl_5(b7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                            count_include_pad=False)
        pool = self.branch_pool(pool)

        return torch.cat([b1, b7, b7dbl, pool], 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        b1 = self.branch1x1(x)

        b3 = self.branch3x3_1(x)
        b3 = [
            self.branch3x3_2a(b3),
            self.branch3x3_2b(b3),
        ]
        b3 = torch.cat(b3, 1)

        b3dbl = self.branch3x3dbl_1(x)
        b3dbl = self.branch3x3dbl_2(b3dbl)
        b3dbl = [
            self.branch3x3dbl_3a(b3dbl),
            self.branch3x3dbl_3b(b3dbl),
        ]
        b3dbl = torch.cat(b3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                            count_include_pad=False)
        pool = self.branch_pool(pool)

        return torch.cat([b1, b3, b3dbl, pool], 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        b1 = self.branch1x1(x)

        b3 = self.branch3x3_1(x)
        b3 = [
            self.branch3x3_2a(b3),
            self.branch3x3_2b(b3),
        ]
        b3 = torch.cat(b3, 1)

        b3dbl = self.branch3x3dbl_1(x)
        b3dbl = self.branch3x3dbl_2(b3dbl)
        b3dbl = [
            self.branch3x3dbl_3a(b3dbl),
            self.branch3x3dbl_3b(b3dbl),
        ]
        b3dbl = torch.cat(b3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling.
        pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        pool = self.branch_pool(pool)

        return torch.cat([b1, b3, b3dbl, pool], 1)