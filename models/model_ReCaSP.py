import torch
import numpy as np
# from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

# from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer
from models.layers.transformer_decoder import TQN_Model
import pdb

import math
import pandas as pd

def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, dim, pooling_type='avg'):
        super(GlobalFeatureExtractor, self).__init__()
        self.pooling_type = pooling_type
        if pooling_type == 'attention':
            self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, num_patches, feature_dim)
            mask: Tensor of shape (batch_size, num_patches), with 1 for masked positions and 0 for unmasked positions

        Returns:
            global_feature: Tensor of shape (batch_size, feature_dim)
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
            # x = x * (1 - mask)

        if self.pooling_type == 'avg':
            if mask is not None:
                valid_mask = (1 - mask).float()
                global_feature = x.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)
            else:
                global_feature = x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        return global_feature


class ReCaSP(nn.Module):
    def __init__(
        self,
        omic_sizes=[100, 200, 300, 400, 500, 600],
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names = [],
        bag_loss = 'nll_surv',
        reg_loss = None,
        pooling_type = 'avg', #'avg', 'max', 或 'attention'
        decoder_number_layer=1,
        ):
        super(ReCaSP, self).__init__()
        self.reg_loss = reg_loss
        self.pooling_type = pooling_type
        self.decoder_number_layer = decoder_number_layer
        self.bag_loss = bag_loss

        #---> general props
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout

        #---> omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        #---> omics props
        self.init_per_path_model(omic_sizes)

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )

        #---> logits props
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )

        input_wsi_dim = self.wsi_projection_dim
        if self.reg_loss == 'dqn_align_loss':
            self.wsi_global_feat_extractor = GlobalFeatureExtractor(input_wsi_dim, pooling_type=self.pooling_type)
            self.omics_global_feat_extractor = GlobalFeatureExtractor(input_wsi_dim, pooling_type=self.pooling_type)
            self.fusion_module = TQN_Model(embed_dim=input_wsi_dim, class_num=1,
                                           decoder_number_layer=self.decoder_number_layer)

        if self.bag_loss == 'trust_nll_surv':
            self.trustClassifier = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim / 4)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(int(self.wsi_projection_dim / 4), self.num_classes * 2),
                nn.Softplus()
            )

    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

    def forward(self, **kwargs):
        wsi_patch_mask = kwargs["wsi_patch_mask"]
        wsi = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,self.num_pathways+1)]
        mask = None
        return_attn = kwargs["return_attn"]

        #---> get pathway embeddings
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer

        h_omic_bag = torch.stack(h_omic)  # omic embeddings are stacked (to be used in co-attention)
        h_omic_bag = h_omic_bag.permute(1, 0, 2)  # (batch_size, num_pathways, 256)

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        if self.reg_loss == 'dqn_align_loss':
            wsi_global_emb = self.wsi_global_feat_extractor(wsi_embed, mask=wsi_patch_mask)
            oimcs_global_emb = self.omics_global_feat_extractor(h_omic_bag, mask=None)
            w2o_cls = self.fusion_module(wsi_embed, oimcs_global_emb).squeeze(-1)
            o2w_cls = self.fusion_module(h_omic_bag, wsi_global_emb).squeeze(-1)
        else:
            w2o_cls = None
            o2w_cls = None

        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        #---> aggregate
        # modality specific mean
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]

        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

        #---> get logits
        logits = self.to_logits(embedding)

        if self.bag_loss == 'trust_nll_surv':
            evidences = self.trustClassifier(embedding)
        else:
            evidences = None

        if return_attn:
            return logits, evidences, w2o_cls, o2w_cls, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits, evidences, w2o_cls, o2w_cls
