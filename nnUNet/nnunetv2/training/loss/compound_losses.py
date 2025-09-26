import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from monai.losses import SoftclDiceLoss



class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

from monai.losses import SoftclDiceLoss
import torch
import torch.nn.functional as F

class DC_and_CE_loss_wClDice(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,
                 ignore_label=None, dice_class=SoftDiceLoss,
                 cldice_weight: float = 0.3, cldice_iter: int = 3, cldice_smooth: float = 1.0):
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label, dice_class)
        from monai.losses import SoftclDiceLoss
        self.cldice_w = float(cldice_weight)
        self._cldice = SoftclDiceLoss(iter_=cldice_iter, smooth=cldice_smooth)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        base = super().forward(net_output, target)
        if self.cldice_w == 0:
            return base

        probs = softmax_helper_dim1(net_output)  # (B,C,...)
        C = probs.shape[1]

        # vectorized one-hot
        if target.ndim == probs.ndim:
            t = target.float()
        else:
            t = torch.zeros_like(probs)
            # target expected as (B,1,...) class indices for nnU-Net CE path
            t.scatter_(1, target.long().clamp_min(0), 1.0)

        if self.ignore_label is not None:
            valid = (target != self.ignore_label).float()
            probs = probs * valid
            t = t * valid

        if C > 1:
            probs = probs[:, 1:]  # drop bg
            t = t[:, 1:]

        return base + self.cldice_w * self._cldice(probs, t)


class DC_and_BCE_loss_wClDice(DC_and_BCE_loss):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1,
                 use_ignore_label: bool = False, dice_class=MemoryEfficientSoftDiceLoss,
                 cldice_weight: float = 0.3, cldice_iter: int = 3, cldice_smooth: float = 1.0):
        super().__init__(bce_kwargs, soft_dice_kwargs, weight_ce, weight_dice, use_ignore_label, dice_class)
        from monai.losses import SoftclDiceLoss
        self.cldice_w = float(cldice_weight)
        self._cldice = SoftclDiceLoss(iter_=cldice_iter, smooth=cldice_smooth)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        base = super().forward(net_output, target)
        if self.cldice_w == 0:
            return base

        probs = torch.sigmoid(net_output)               # (B,Cp,...), Cp is model output channels
        if self.use_ignore_label:
            mask = ~target[:, -1:].bool() if target.dtype == torch.bool else (1 - target[:, -1:]).bool()
            tgt = target[:, :-1].float()                # drop ignore channel from TARGET only
            # broadcast-safe mask
            probs = probs * mask
            tgt = tgt * mask
        else:
            tgt = target.float()

        assert probs.shape[1] > 0, f"clDice got zero channels: probs={tuple(probs.shape)}"
        assert probs.shape[1] == tgt.shape[1], f"pred/target channel mismatch for clDice: {probs.shape} vs {tgt.shape}"

        return base + self.cldice_w * self._cldice(probs, tgt)


