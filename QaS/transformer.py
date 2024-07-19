# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transformer class.

"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from module import VideoEncoder, TextEncoder, CrossDecoder



class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_props=8, num_neg_props=8, num_decoder_layers1=3,
                 num_decoder_layers2=3, num_concept=1, frames_input_size=2048, words_input_size=300,
                 dropout=0.1, sigma=9, gamma=0.5, vocab_size=None, use_negative=True, neg_type=None, 
                 num_patterns=0, max_epoch=30
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_patterns = num_patterns
        self.num_props=num_props
        self.num_concepts=num_concept
        self.num_neg_props=num_neg_props
        self.sigma=sigma
        self.gamma=gamma
        self.dropout = dropout
        self.max_epoch=max_epoch
        self.vocab_size=vocab_size
        self.use_negative=use_negative
        self.neg_type=neg_type
        
        self.fc_gauss = nn.Linear(d_model, num_props*2)
        self.neg_fc_gauss = nn.Linear(d_model, num_props*2)
        self.frame_fc = nn.Linear(frames_input_size, d_model)
        self.word_fc = nn.Linear(words_input_size, d_model)
        self.fc_comp = nn.Linear(d_model, self.vocab_size)
        
        self.start_vec = nn.Parameter(torch.zeros(words_input_size).float(), requires_grad=True)
        self.vid_pred_vec = nn.Parameter(torch.zeros(frames_input_size).float(), requires_grad=True)
        self.txt_pred_vec = nn.Parameter(torch.zeros(words_input_size).float(), requires_grad=True)
        self.mask_vec = nn.Parameter(torch.zeros(words_input_size).float(), requires_grad=True)

        self.vid_encoder = VideoEncoder(d_model, nhead, num_decoder_layers1, num_decoder_layers2, concept_nums=num_concept)
        self.txt_encoder = TextEncoder(d_model, nhead, num_decoder_layers1, num_decoder_layers2, concept_nums=num_concept)
        self.cross_decoder = CrossDecoder(d_model, nhead, num_decoder_layers1, num_decoder_layers2, concept_nums=num_concept)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(d_model, 0, 20)



    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        
        bsz, n_frames, _ = frames_feat.shape
        device = frames_feat.device
        vid_pred_vec = self.vid_pred_vec.view(1, 1, -1).expand(bsz, 1, -1)

        frames_feat = torch.cat([frames_feat, vid_pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len + 1)
        frames_mask = frames_mask.to(device)
        src_vid_mask = frames_mask[:, :-1]

        words_feat[:, 0] = self.start_vec.to(device)
        words_feat[:, -1] = self.txt_pred_vec.to(device)
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)
        words_mask = words_mask.to(device)
        words_mask[:,-1] = 1

        # pos_vid_concept, hy = self.vid_encoder(frames_feat, frames_mask, gauss_weight=None) # w/o fusion
        pos_vid_concept, hy = self.vid_encoder(frames_feat, frames_mask, words_feat + words_pos, words_mask, gauss_weight=None, require_enc=False) 
        memory_local = hy[:, 1:, :]
        h = self.fc_gauss(pos_vid_concept).reshape(bsz, self.num_props, 2).sigmoid()
        gauss_center = h[:, :, 0].reshape(-1)
        gauss_width = h[:, :, 1].reshape(-1)
        if self.neg_type == 'lnb':
            n_h = self.neg_fc_gauss(pos_vid_concept).reshape(bsz, self.num_props, 2).sigmoid()
            neg_gauss_center = n_h[:, :, 0].reshape(-1)
            neg_gauss_width = n_h[:, :, 1].reshape(-1)
        else:
            neg_gauss_center, neg_gauss_width = None, None
        
        # downsampling
        props_len = int(n_frames // 1.5)
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        frames_feat = torch.cat((frames_feat[:, keep_idx], frames_feat[:,-2:-1]), dim=1)
        frames_mask = torch.cat((frames_mask[:, keep_idx], frames_mask[:,-2:-1]), dim=1)
        props_len += 1
        
        gauss_weight = self.generate_gauss_weight(props_len-1, gauss_center, gauss_width)
        gauss_weight = torch.cat((gauss_weight, torch.zeros((gauss_weight.shape[0], 1)).type_as(gauss_weight)),  dim=-1)
        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]

        props_feat = frames_feat.unsqueeze(1).expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1).expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        
        masked_words_feat, _ = self._mask_words(words_feat, words_len, weights=weights)
        masked_words_feat = masked_words_feat + words_pos
        masked_words_feat = masked_words_feat[:, :-2]
        masked_words_mask = words_mask[:, :-2]

        pos_vid_concept, _ = self.vid_encoder(props_feat, props_mask, gauss_weight=pos_weight)
        word_concept = self.txt_encoder(words_feat, words_mask)

        
        h, cross_h = self.cross_decoder(pos_vid_concept, None, word_concept, None, masked_words_feat, masked_words_mask)
        words_logit = self.fc_comp(h)
        cross_words_logit = self.fc_comp(cross_h)
        

       
        if self.use_negative and self.training:
            if self.neg_type == 'rev':
                neg_1_weight = 1.0 - pos_weight
            elif self.neg_type == 'lnb':
                neg_1_weight = self.generate_gauss_weight(props_len-1, neg_gauss_center, neg_gauss_width)
                neg_1_weight = torch.cat((neg_1_weight, torch.zeros((neg_1_weight.shape[0], 1)).type_as(gauss_weight)), dim=-1)
            neg_vid_concept_1, _ = self.vid_encoder(props_feat, props_mask, gauss_weight=neg_1_weight)
            _, neg_cross_h_1 = self.cross_decoder(neg_vid_concept_1, None, tgt = masked_words_feat, tgt_mask = masked_words_mask)
            neg_words_logit_1 = self.fc_comp(neg_cross_h_1)

            ref_concept, _ = self.vid_encoder(frames_feat, frames_mask)
            _, ref_cross_h = self.cross_decoder(ref_concept, None, tgt = masked_words_feat, tgt_mask = masked_words_mask)
            ref_words_logit = self.fc_comp(ref_cross_h)
            

        else:
            neg_vid_concept_1 = None
            neg_words_logit_1 = None
            ref_concept = None
            ref_words_logit = None
        

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': None,
            'pos_words_logit': cross_words_logit,
            'ref_words_logit':  ref_words_logit,

            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': masked_words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,

            'pos_vid_concepts': pos_vid_concept,
            'neg_vid_concepts_1': neg_vid_concept_1,
            'neg_vid_concepts_2': None,
            'ref_concept': ref_concept,
            'txt_concepts': word_concept,

            'hy': hy,
            'video_mask': src_vid_mask,
            'n_width': neg_gauss_width,
            'n_center': neg_gauss_center
        }

    
    def generate_gauss_weight(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]
    
    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.to(words_feat.device).unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)


        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1)
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().to(words_feat.device))
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1) # (bsz, Lq, 1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_props=args.num_props,
        num_concept=args.num_concepts,
        num_neg_props=args.num_neg_props,
        frames_input_size=args.v_feat_dim,
        words_input_size=args.t_feat_dim,
        sigma=args.sigma,
        gamma=args.gamma,
        vocab_size=args.vocab_size,
        num_decoder_layers1=args.dec_layers1,
        num_decoder_layers2=args.dec_layers2,
        use_negative=args.use_negative,
        neg_type=args.neg_type
    )
