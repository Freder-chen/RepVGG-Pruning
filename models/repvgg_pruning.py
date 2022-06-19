import copy
import numpy as np

import torch
import torch.nn as nn


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
    return nn.Sequential(
        ('conv', nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )),
        ('bn', nn.BatchNorm2d(num_features=out_channels))
    )


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(RepVGGBlock, self).__init__()
        # self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        self.rbr_1x1 = conv_bn(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
            stride=stride, padding=padding_11, dilation=dilation, groups=groups
        )
        # print('RepVGG Block, identity = ', self.rbr_identity)
        
        self.running = nn.BatchNorm2d(out_channels, affine=False)
        self.mask = nn.Conv2d(out_channels, out_channels, 1, groups=out_channels, bias=False)
        nn.init.ones_(self.mask.weight)

    def forward(self, inputs):
        # deploy states
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
        
        # train states
        id_out = self.rbr_identity(inputs) if self.rbr_identity is not None else 0

        outputs = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        self.running(outputs)
        outputs = self.mask(outputs)

        return self.nonlinearity(outputs)
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = conv_bn(
            self.rbr_dense.conv.in_channels,
            self.rbr_dense.conv.out_channels,
            self.rbr_dense.conv.kernel_size,
            self.rbr_dense.conv.stride,
            self.rbr_dense.conv.padding,
            self.rbr_dense.conv.dilation,
            self.rbr_dense.conv.groups,
            bias=False,
        )

        # reparam conv
        self.rbr_reparam.conv.weight.data = kernel
        # self.rbr_reparam.conv.bias.data = bias  # conv bias to bn bias

        # reparam bn
        bn_var_sqrt = torch.sqrt(self.running.running_var + self.running.eps)
        self.rbr_reparam.bn.weight.data = bn_var_sqrt
        self.rbr_reparam.bn.bias.data = self.running.running_mean + bias
        self.rbr_reparam.bn.running_mean.data = self.running.running_mean
        self.rbr_reparam.bn.running_var.data = self.running.running_var
        # mask bn weight
        self.rbr_reparam.bn.weight.data *= self.mask.weight.data.reshape(-1)
        self.rbr_reparam.bn.bias.data *= self.mask.weight.data.reshape(-1)

        # reparam grad
        for para in self.parameters():
            para.detach_()

        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.__delattr__('running')
        self.__delattr__('mask')

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return torch.nn.functional.pad(kernel1x1, [1,1,1,1]) if kernel1x1 is not None else 0

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, in_channels=3):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.in_channels = in_channels

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(
            in_channels=self.in_channels, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1
        )
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(
                in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=1,
            ))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def update_mask(self, sr, threshold):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1,1) and m.groups != 1:
                    m.weight.grad.data.add_(sr * torch.sign(m.weight.data))
                    # After death, no longer active
                    m1 = m.weight.data.abs() > threshold
                    m.weight.grad.data *= m1  # if weight < threshold: grad = 0
                    m.weight.data *= m1   # if weight < threshold: weight = 0

    def fix_mask(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1) and m.groups != 1:
                    m.weight.requires_grad = False
    
    def deploy_repvgg(self):
        model = copy.deepcopy(self)
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        return model

    def prune(self):
        in_mask = torch.ones(self.in_channels) > 0
        model = self.deploy_repvgg()
        
        prune_cnt = 0; total_cnt = 0
        for m in model.modules():
            if isinstance(m, RepVGGBlock):
                # use static thresh, thresh = 0.
                mask = m.rbr_reparam.bn.weight.data.abs().reshape(-1) > 0.0001

                prune_cnt += int(mask.sum())
                total_cnt += int(len(mask))

                # prune model
                temp_param = conv_bn(
                    int(in_mask.sum()),
                    int(mask.sum()),
                    m.rbr_reparam.conv.kernel_size,
                    m.rbr_reparam.conv.stride,
                    m.rbr_reparam.conv.padding,
                    m.rbr_reparam.conv.dilation,
                    m.rbr_reparam.conv.groups,
                    bias=False
                )
                temp_param.conv.weight.data = m.rbr_reparam.conv.weight.data[mask][:, in_mask]
                temp_param.bn.weight.data = m.rbr_reparam.bn.weight.data[mask]
                temp_param.bn.bias.data = m.rbr_reparam.bn.bias.data[mask]
                temp_param.bn.running_mean = m.rbr_reparam.bn.running_mean[mask]
                temp_param.bn.running_var = m.rbr_reparam.bn.running_var[mask]
                m.rbr_reparam = temp_param

                in_mask = mask  # next input mask
            
            elif isinstance(m, nn.Linear):
                linear = nn.Linear(int(in_mask.sum()), m.out_features)
                linear.weight.data = m.weight.data[:, in_mask]
                linear.bias.data = m.bias.data
                model.linear = linear
        
        print(f'=> pruning ratio: {1 - prune_cnt / total_cnt}')
        
        return model


def _repvgg(arch, num_blocks, num_classes, width_multiplier, pretrained):
    model = RepVGG(
        num_blocks=num_blocks, num_classes=num_classes, width_multiplier=width_multiplier, in_channels=3
    )

    # TODO: load pretrain state_dict
    # if pretrained:
    #     checkpoint_model = torch.load(model_urls[arch], map_location=torch.device('cpu'))
    #     state_dict = model.state_dict()
    #     for k in ['linear.weight', 'linear.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #     model.load_state_dict(checkpoint_model, strict=False)
    #     print("success load pretrained model")

    return model


def repvgg_a0(num_classes=1000, pretrained=False):
    return _repvgg(
        'repvgg_a0', num_blocks=[2, 4, 14, 1], num_classes=num_classes,
        width_multiplier=[0.75, 0.75, 0.75, 2.5], pretrained=pretrained
    )

def repvgg_a1(num_classes=1000, pretrained=False):
    return _repvgg(
        'repvgg_a1', num_blocks=[2, 4, 14, 1], num_classes=num_classes,
        width_multiplier=[1, 1, 1, 2.5], pretrained=pretrained
    )

def repvgg_a2(num_classes=1000, pretrained=False):
    return _repvgg(
        'repvgg_a2', num_blocks=[2, 4, 14, 1], num_classes=num_classes,
        width_multiplier=[1.5, 1.5, 1.5, 2.75], pretrained=pretrained
    )

def repvgg_b0(num_classes=1000, pretrained=False,):
    return _repvgg(
        'repvgg_b0', num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[1, 1, 1, 2.5], pretrained=pretrained
    )

def repvgg_b1(num_classes=1000, pretrained=False):
    return _repvgg(
        'repvgg_b1', num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2, 2, 2, 4], pretrained=pretrained
    )

def repvgg_b2(num_classes=1000, pretrained=False):
    return _repvgg(
        'repvgg_b2', num_blocks=[4, 6, 16, 1], num_classes=num_classes,
        width_multiplier=[2.5, 2.5, 2.5, 5], pretrained=pretrained
    )
