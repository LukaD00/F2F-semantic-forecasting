import torch
import torch.nn as nn
from itertools import chain
from .util import _BNReluConv, upsample


class ScaleInvariantModel(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(ScaleInvariantModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)
        self.criterion = None

    def forward(self, pyramid, mux_masks, target_size, image_size):
        feats, additional = zip(*[self.backbone(p) for p in pyramid])
        feature_pyramid = [upsample(f, target_size) for f in feats]
        features = feature_pyramid[0] if len(feature_pyramid) == 1 else None
        logits = self.logits.forward(features)
        return upsample(logits, image_size), additional[0]

    def forward_down(self, x):
        return self.backbone.forward_down(x)

    def forward_up(self, feats, target_size, image_size):
        feats, additional = self.backbone.forward_up(feats)
        features = upsample(feats, target_size)
        return upsample(self.logits.forward(features), image_size), additional

    def forward_upsample_only(self, feats):
        return self.backbone.forward_up(feats)[0]

    def forward_up_without_bbup(self, feats, target_size, image_size):
        features = upsample(feats, target_size)
        return upsample(self.logits.forward(features), image_size), 0

    def prepare_data(self, batch, image_size, device=torch.device('cuda')):
        if image_size is None:
            image_size = batch['target_size']
        req_grad = self.calculate_receptive_field
        pyramid = [p.clone().detach().requires_grad_(req_grad).to(device) for p in batch['pyramid']]
        mux_masks = [mi.to(device) for mi in batch['mux_masks']] if 'mux_masks' in batch else None
        return {
            'pyramid': pyramid,
            'mux_masks': mux_masks,
            'image_size': image_size,
            'target_size': batch['target_size_feats']
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        return self.forward(**data)

    def loss(self, batch):
        assert self.criterion is not None
        labels = batch['labels'].cuda()
        logits, _ = self.do_forward(batch, image_size=labels.shape[-2:])
        return self.criterion(logits, labels)

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()