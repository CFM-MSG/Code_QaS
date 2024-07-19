# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn.functional as F
from torch import nn

from qas.transformer import build_transformer
from qas.loss_utils import cal_nll_loss

class QaS(nn.Module):

    def __init__(self, transformer, num_props, max_v_l=75, use_txt_pos=False, n_input_proj=2):
        """ Initializes the model.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.max_v_l = max_v_l
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.num_props = num_props
        self.num_concepts=transformer.num_concepts

        self.sal_fc = nn.Linear(hidden_dim, 1)

        self.hidden_dim = hidden_dim


    def forward(self, src_txt, words_id, weights, words_len, src_vid, src_vid_mask, **kwargs):
       
        bsz, video_length, _ = src_vid.shape
        video_length = torch.tensor(video_length, dtype=int).repeat(bsz)

        outputs = self.transformer(src_vid, video_length, words_id, src_txt, words_len, weights)
 
        # TODO saliency branch
        hy, cross_words_logit, masked_words_mask = outputs['hy'], outputs['pos_words_logit'], outputs['words_mask']
        words_id1, gauss_center, gauss_width = outputs['words_id'], outputs['center'], outputs['width']
        neg_gauss_center, neg_gauss_width = outputs['n_center'], outputs['n_width']
        neg_words_logit_1 = outputs['neg_words_logit_1']
        saliency_score, pred_spans, neg_spans_left, neg_spans_right = \
            self.saliency_branch(hy, cross_words_logit, neg_words_logit_1, masked_words_mask, words_id1, gauss_center, gauss_width, neg_gauss_center, neg_gauss_width)
        
        outputs['saliency_score'] = saliency_score
        outputs['pred_spans'] = pred_spans
        outputs['neg_spans_left'] = neg_spans_left
        outputs['neg_spans_right'] = neg_spans_right

        return outputs

    def saliency_branch(self, memory_local, words_logit, neg_words_logit_1, words_mask, words_id, gauss_center, gauss_width, neg_gauss_center=None, neg_gauss_width=None):
        n_frames = memory_local.shape[1]
        bsz = words_logit.shape[0] // self.num_props
        gauss_center, gauss_width = gauss_center.reshape(bsz, -1), gauss_width.reshape(bsz, -1)
        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)

        pos_nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
        pos_nll_loss = pos_nll_loss.view(bsz, self.num_props)
        pos_rank_idx = torch.sort(pos_nll_loss, dim=1)[1][:, :1] # bsz, 1
        center = torch.gather(gauss_center, dim=1, index=pos_rank_idx) # bsz, 1
        width = torch.gather(gauss_width, dim=1, index=pos_rank_idx)

        pred_spans = torch.cat([torch.clamp(center-width/2, min=0), 
                                  torch.clamp(center+width/2, max=1)], dim=-1) # bsz, 2
        pred_spans = (pred_spans * n_frames).round().int()

        neg_spans_left = torch.zeros(bsz, 2).int().to(words_logit.device)
        neg_spans_right = torch.zeros(bsz, 2).int().to(words_logit.device)

        if neg_gauss_center is not None and neg_gauss_width is not None and neg_words_logit_1 is not None:
            neg_nll_loss, neg_acc = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
            neg_nll_loss = neg_nll_loss.view(bsz, self.num_props)
            neg_nll_idx = torch.sort(neg_nll_loss, descending=True, dim=1)[1][:, :1] # (bsz, )
            neg_width = torch.gather(neg_gauss_width.reshape(bsz, self.num_props), dim=1, index=neg_nll_idx)
            neg_center = torch.gather(neg_gauss_center.reshape(bsz, self.num_props), dim=1, index=neg_nll_idx)
            neg_spans = torch.cat([torch.clamp(neg_center-neg_width/2, min=0), 
                                torch.clamp(neg_center+neg_width/2, max=1)], dim=-1) # bsz, 2
            neg_spans_left = (neg_spans * n_frames).round().int()
            neg_spans_right = None
        else:
            neg_spans_right[:, 1] = n_frames
            neg_spans_left[:, 1] = pred_spans[:, 0]
            neg_spans_right[:, 0] = pred_spans[:, 1]
        
        saliency_score = self.sal_fc(memory_local).squeeze(-1)#.sigmoid()

        return saliency_score, pred_spans, neg_spans_left, neg_spans_right


def build_model(args):
    transformer = build_transformer(args)

    model = QaS(
        transformer,
        num_props=args.num_props,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )
    
    return model

