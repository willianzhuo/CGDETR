# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
CG-DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from cg_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from cg_detr.matcher import build_matcher
from cg_detr.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from cg_detr.position_encoding import build_position_encoding
from cg_detr.misc import accuracy
import numpy as np
import copy

# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res

class CGDETR(nn.Module):
    """ CG DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0, args=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         CG-DETR can detect in a single video. 控制最终输出的预测moment的数量
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.args=args
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3) # 用于预测moment的位置
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.token_type_embeddings = nn.Embedding(2, hidden_dim) # 用于区分video和text用，设置嵌入空间的维度为2，即将视频和文本分别嵌入到2维空间中，以便后面输入到transformer中可以区分
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj # number of layers for input projection控制线性层的数量
        self.query_embed = nn.Embedding(num_queries, 2) # 控制最终输出的预测moment的数量 用于生成查询嵌入
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        # self.query_proj = nn.Sequential(*[
        #     LinearLayer(hidden_dim, 128, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
        #     LinearLayer(128, 64, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
        #     LinearLayer(64, 16, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
        #     LinearLayer(16, 4, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
        #     LinearLayer(4, 2, layer_norm=True, dropout=input_dropout, relu=relu_args[1])
        # ])
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        # Sec. 3.3
        self.contrastive_align_loss = contrastive_align_loss #是否使用对比对齐损失
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.aux_loss = aux_loss #是否使用辅助损失 默认使用
        self.hidden_dim = hidden_dim
        
        # Saliency Candidates Pool Figure 5
        self.global_rep_token = torch.nn.Parameter(torch.randn(args.total_prompts, hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.moment_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.moment_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim)) # dummy token qv中默认45个
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        self.sent_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.sent_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        self.txt_proj_linear = LinearLayer(txt_dim, hidden_dim, layer_norm=True)

        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm) # Sec. 3.2 Dummy Encoder

        scls_encoder_layer = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        scls_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.scls_encoder = TransformerEncoder(scls_encoder_layer, args.sent_layers, scls_encoder_norm) # Sec. 3.3 sentence token encoder Figure 4 (a) 

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, src_aud=None, src_aud_mask=None, targets=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer 表示的是query的长度
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer 表示的是video的长度

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        ## For discovering real negative samples
        if vid is not None: ## for demo (run_on_video/run.py)
            _count = [v.count('_') for v in vid]
            if self.args.dset_name == 'hl':
                _position_to_cut = [find_nth(v, '_', _count[i]-1) for i, v in enumerate(vid)]
                ori_vid = [v[:_position_to_cut[i]] for i, v in enumerate(vid)]
            else:
                ori_vid = [v for v in vid]

        if src_aud is not None: # 是否有音频部分
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        # Project inputs to the same hidden dimension
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt) # torch.Size([32, 28, 256])
        query_proj = self.query_proj(src_txt) # torch.Size([32, 28, 2])
        # query_proj = torch.mean(query_proj, dim=1) # torch.Size([32, 2])
        # Add type embeddings
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        # Add position embeddings
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d) 默认不用

        ######Sec. 3.2 insert dummy token in front of txt 
        txt_dummy = self.dummy_rep_token.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1) #创建一个新的张量 txt_dummy，它的每一行都是 self.dummy_rep_token 的一个副本，总共有 src_txt.shape[0] 行
        src_txt_dummy = torch.cat([txt_dummy, src_txt], dim=1) # 把dummy token插入到src_txt的前面


        mask_txt = torch.tensor([[True] * self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1) # [32,45] 初始dummy的mask
        src_txt_mask_dummy = torch.cat([mask_txt, src_txt_mask], dim=1) # [32,68] 把dummy的mask插入到src_txt_mask的前面

        pos_dummy = self.dummy_rep_pos.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1) # [32,45,256]创建一个新的张量 pos_dummy，它的每一行都是 self.dummy_rep_pos 的一个副本，总共有 pos_txt.shape[0] 行
        pos_txt_dummy = torch.cat([pos_dummy, pos_txt], dim=1) # [32,68,256] 把dummy的pos插入到pos_txt的前面
        src_txt_dummy = src_txt_dummy.permute(1, 0, 2)  # (L, batch_size, d)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)   # (L, batch_size, d)

        memory = self.txtproj_encoder(src_txt_dummy, src_key_padding_mask=~(src_txt_mask_dummy.bool()), pos=pos_txt_dummy)  # (L, batch_size, d) Sec. 3.2 Dummy Encoder 的输出
        dummy_token = memory[:self.args.num_dummies].permute(1, 0, 2) # Sec. 3.2 Dummy Encoder 的输出截取出的dummy token
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)  # (batch_size, L, d) 还原pos_txt_dummy的形状

        src_txt_dummy = torch.cat([dummy_token, src_txt], dim=1) # torch.Size([32, 68, 256])
        mask_txt_dummy = torch.tensor([[True] * self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1) # [32,45] 3.2encoder之后的dummy的mask
        src_txt_mask_dummy = torch.cat([mask_txt_dummy, src_txt_mask], dim=1)

        # Sec. 3.2 Adaptive Cross-Attention 的 Input : Concat video, dummy, txt
        src = torch.cat([src_vid, src_txt_dummy], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask_dummy], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1) # torch.Size([32, 143, 256])

        # for t2vidavg sal token
        vidsrc_ = torch.zeros((len(src_vid), 1, self.hidden_dim)).cuda() # [32,1,256]
        for i in range(len(src_vid)):
            vidsrc_[i] = src_vid[i][:src_vid_mask.sum(1)[i].long()].mean(0).clone().detach() # [32,1,256] 用视频的均值作为sal token (计算有效元素的均值，然后将结果存储在 vidsrc_ 的第 i 个元素)

        video_length = src_vid.shape[1] # 75
        if targets is not None: ## train
            hs, reference, memory, memory_global, attn_weights = self.transformer(src, ~mask, query_proj, pos, video_length=video_length, moment_idx=targets["relevant_clips"],
                                                                                                                  ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())
        else: ## inference
            hs, reference, memory, memory_global, attn_weights = self.transformer(src, ~mask, query_proj, pos, video_length=video_length,
                                                                                                                  ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes) [3, 32, 10, 2] 将隐藏层的输出hs输入到全连接层中，得到输出outputs_class
        reference_before_sigmoid = inverse_sigmoid(reference) # (#layers, batch_size, #queries, 2) [3, 32, 10, 2] 将reference的输出输入到inverse_sigmoid中，得到输出reference_before_sigmoid
        tmp = self.span_embed(hs) # (#layers, batch_size, #queries, 2) [3, 32, 10, 2] 将隐藏层的输出hs输入到全连接层中，得到输出tmp
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]} # 只取最后一层的输出

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d) [32, 0, 256]
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d) [32, 75, 256]
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        if vid is not None: ## for demo (run_on_video/run.py)
            ### Neg Pairs ###
            neg_vid = ori_vid[1:] + ori_vid[:1] #将 ori_vid 列表的第一个元素移动到最后一个位置
            real_neg_mask = torch.Tensor(element_wise_list_equal(ori_vid, neg_vid)).to(src_txt_dummy.device)
            real_neg_mask = real_neg_mask == False
            if real_neg_mask.sum() != 0:

                src_txt_dummy_neg = torch.cat([src_txt_dummy[1:], src_txt_dummy[0:1]], dim=0)
                src_txt_mask_dummy_neg = torch.cat([src_txt_mask_dummy[1:], src_txt_mask_dummy[0:1]], dim=0)
                src_dummy_neg = torch.cat([src_vid, src_txt_dummy_neg], dim=1)
                mask_dummy_neg = torch.cat([src_vid_mask, src_txt_mask_dummy_neg], dim=1).bool()
                pos_neg = pos.clone()  # since it does not use actual content

                mask_dummy_neg = mask_dummy_neg[real_neg_mask] # 挑选出neg mask中为True的位置 [32, 143]
                src_dummy_neg = src_dummy_neg[real_neg_mask] # [32, 143, 256]
                pos_neg = pos_neg[real_neg_mask]
                src_txt_mask_dummy_neg = src_txt_mask_dummy_neg[real_neg_mask]
                query_proj = query_proj[real_neg_mask]

                _, _, memory_neg, memory_global_neg, attn_weights_neg = self.transformer(src_dummy_neg, ~mask_dummy_neg, query_proj, pos_neg, video_length=video_length,
                                                                                               ctxtoken=vidsrc_[real_neg_mask], gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask[real_neg_mask].sum(1).long())
                vid_mem_neg = memory_neg[:, :src_vid.shape[1]] # (bsz, L_vid, d)
                out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
                out["src_txt_mask_neg"] = src_txt_mask_dummy_neg

                out["t2vattnvalues_neg"] = (attn_weights_neg[:, :, self.args.num_dummies:] * (src_txt_mask_dummy_neg[:, self.args.num_dummies:].unsqueeze(1).repeat(1, video_length, 1))).sum(2)
                out["t2vattnvalues_neg"] = torch.clamp(out["t2vattnvalues_neg"], 0, 1) # [32, 75]
            else:
                out["saliency_scores_neg"] = None
                out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = real_neg_mask
        else:
            out["saliency_scores_neg"] = None
            out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = None


        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim)) # [32, 75, 256] * [32, 256] / sqrt(256) -> [32, 75]


        out["t2vattnvalues"] = (attn_weights[:,:,self.args.num_dummies:] * (src_txt_mask.unsqueeze(1).repeat(1, video_length, 1))).sum(2) # (batch_size, L_vid, L_txt) / (batch_size, L_txt)
        out["t2vattnvalues"] = torch.clamp(out["t2vattnvalues"], 0, 1)
        out["dummy_tokens"] = dummy_token
        out["global_rep_tokens"] = self.global_rep_token

        out["video_mask"] = src_vid_mask
        ## 根据query token和moment token（=原始clip token求平均）计算query和预测区间的相似度分数
        # 计算出预测区间包含的clip的token的平均值
        # 找出pred_spans中的每个区间的clip
        # pred_spans = out["pred_spans"]
        # moment_token = []
        # for i in range(pred_spans.shape[0]):
        #     start = pred_spans[i, :, 0].long()
        #     end = pred_spans[i, :, 1].long()
        #     clips = src_vid[i, start:end+1, :]
        #     moment_token.append(clips)
        # moment_token = torch.stack(moment_token, dim=0)
        # moment_token = src_vid[:, out["pred_spans"][..., 1] - out["pred_spans"][..., 0], :].mean(dim=1)
        
        out["query_token"] = src_txt.mean(dim=1) # torch.Size([32, 256])
        out["vid_token"] = src_vid
        
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out


class SAMMIMIC(nn.Module):
    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0, args=None):        
        super().__init__()
        self.transformer = transformer
        self.num_queries = num_queries
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        self.txt_dim = txt_dim
        self.vid_dim = vid_dim
        hidden_dim = transformer.d_model
        self.aux_loss = aux_loss
        normalize_before = False
        self.contrastive_align_loss = contrastive_align_loss
        self.contrastive_hdim = contrastive_hdim
        self.max_v_l = max_v_l
        self.span_loss_type = span_loss_type
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.aud_dim = aud_dim
        self.args = args

        # Video Transformer Encoder
        self.video_encoder_layer = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu") #TODO
        self.video_encoder = TransformerEncoder(self.video_encoder_layer, num_layers=args.num_encoder_layers)

        # Text Transformer Encoder
        self.text_encoder_layer = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu")
        self.text_encoder = TransformerEncoder(self.text_encoder_layer, num_layers=args.num_encoder_layers)

        # Transformer Decoder
        self.decoder_layer = TransformerDecoderLayer(d_model=256, nhead=2, dropout=args.dropout)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=3)

        # Project video and text input to the same hidden dimension
        self.video_input_proj = nn.Linear(vid_dim, hidden_dim)
        self.text_input_proj = nn.Linear(txt_dim, hidden_dim)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, src_aud=None, src_aud_mask=None, targets=None):
        # Project inputswo tensors:
        src_vid = self.video_input_proj(src_vid) # bsz, L_vid, d torch.Size([32, 75, 2306]) -> torch.Size([32, 75, 256])
        src_txt = self.text_input_proj(src_txt) # bsz, L_txt, d torch.Size([32, 25, 512]) -> torch.Size([32, 25, 256])

        # Add position embeddings
        # src_vid = src_vid + self.position_embed(src_vid_mask)
        # src_txt = src_txt + self.txt_position_embed(src_txt_mask)
        pos_vid = self.position_embed(src_vid, src_vid_mask) # (bsz, L_vid, d) mask: (bsz, L_vid)(32,75) pos: (bsz, L_vid, d) torch.Size([32, 75, 256])
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d) torch.Size([32, 25, 256])

        # Pass through the encoders
        video_memory = self.video_encoder(pos_vid) # (bsz, L_vid, d) torch.Size([32, 75, 256])
        text_memory = self.text_encoder(pos_txt) # 形状[32, 25, 256]

        print("text_memory shape:", text_memory.shape)
        print("video_memory shape:", video_memory.shape)
        
        # Decoder (cross attention between text query and video key/value)
        output = self.decoder(tgt=text_memory.permute(1,0,2), memory=video_memory.permute(1,0,2))
        return output

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True, args=None):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.args=args
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        # for tvsum,
        self.use_matcher = use_matcher

        # moment sentence contrastive
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.args.device)
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none').to(self.args.device)
        self.bce_criterion = nn.BCELoss(reduction='none')

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        # Neg pair loss
        if outputs["saliency_scores_neg"] is not None: ## When batch size is not 1 (negative pair exists)
            vid_token_mask = outputs["video_mask"]
            real_neg_mask = outputs["real_neg_mask"]
            saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
            loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair * 0.
            else:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
                
            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_mask"]
            # Neg pair loss

            if outputs["t2vattnvalues_neg"] is not None:
                saliency_scores_neg = outputs["t2vattnvalues_neg"].clone()  # (N, L)
                loss_neg_pair_attn = (- torch.log(1. - saliency_scores_neg) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat(
                [saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (
                        1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive_attn = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive_attn = loss_rank_contrastive_attn + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive_attn / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn * 0 + loss_saliency_attn
            else:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn + loss_saliency_attn
            loss_saliency += (loss_saliency_attn * self.args.lw_wattn)
            
        else: ## when batch size == 1
            vid_token_mask = outputs["video_mask"]
            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)  # no drop
                pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            loss_saliency = loss_saliency + loss_rank_contrastive
            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_mask"]
            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)  # no drop
                pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive / 12

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_saliency_attn # 分别对应文章中的L_{rctl}, L_{bce}, L_{saliency}
            loss_saliency += (loss_saliency_attn * self.args.lw_wattn)
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_moment_sentence(self, outputs, targets, indices, log=True):
        if outputs["memory_moment"] is not None:
            moment_token = outputs["memory_moment"]
            nmmemory_moment = outputs["nmmemory_moment"]
            sentence_token = outputs["sentence_txt"].squeeze(1)
            sentence_dummy = outputs["sentence_dummy"].squeeze(1) # b, 1, d

            moment_logits = F.normalize(moment_token, dim=1)
            nmoment_logits = F.normalize(nmmemory_moment, dim=1)
            sentence_logits = F.normalize(sentence_token, dim=1)
            dummy_logits = F.normalize(sentence_dummy, dim=1)

            similarity_matrix = torch.matmul(moment_logits, sentence_logits.T) # B B
            nsimilarity_matrix = torch.matmul(nmoment_logits, sentence_logits.T) # B B
            similarity_matrix = torch.cat([similarity_matrix, nsimilarity_matrix], dim=1)
            labels = torch.eye(similarity_matrix.shape[0]).to(self.args.device)
            nlabels = torch.zeros_like(nsimilarity_matrix).to(self.args.device)
            labels = torch.cat([labels, nlabels], dim=1).max(dim=1)[1]

            loss_ms_align = self.criterion(similarity_matrix, labels)

            dummy_similarity_matrix = torch.matmul(moment_logits, dummy_logits.T)
            dummy_nsimilarity_matrix = torch.matmul(nmoment_logits, dummy_logits.T)
            dummy_similarity_matrix = torch.cat([dummy_similarity_matrix, dummy_nsimilarity_matrix], dim=1)
            dummy_labels = (~(torch.eye(similarity_matrix.shape[0]).to(self.args.device).bool())).float()
            dummy_nlabels = torch.ones_like(nsimilarity_matrix).to(self.args.device)
            dummy_labels = torch.cat([dummy_labels, dummy_nlabels], dim=1).max(dim=1)[1]

            dummy_loss_ms_align = self.criterion(dummy_similarity_matrix, dummy_labels)
            loss_ms_align += dummy_loss_ms_align
            video_mask = outputs['video_mask']
            src_vid = outputs['src_vid']  # [bsz, L_vid, D_vid]
            moment_mask_ = torch.clamp(targets["relevant_clips"], 0, 1)

            momtokcls_pred = torch.matmul(moment_token.unsqueeze(1), src_vid.permute(0, 2, 1))  # bsz 1 L_vid
            momtokcls_label = moment_mask_
            momtokcls_logit = torch.sigmoid(momtokcls_pred)
            loss_ms_align += (self.bce_criterion(momtokcls_logit.reshape(-1), momtokcls_label.reshape(-1)) * video_mask.reshape(-1)).mean()

        else:
            loss_ms_align = 0.
        return {"loss_ms_align": loss_ms_align}
        #

    def loss_moment2txt_sim_distill(self, outputs, targets, indices, log=True):
        if outputs["moment2txt_similarity"] is not None:
            moment2txt_similarity = outputs["moment2txt_similarity"]  # bsz L_clip 22
            moment_mask = outputs["moment_mask"].int() # bsz L_clip 1
            txt_mask = outputs["txt_mask"].unsqueeze(1).repeat(1, outputs["cate_attn_weights"].size(1), 1)  # bsz l_t

            attn_weights = outputs["cate_attn_weights"] # bsz L_clip 22
            b, L_vid, L_txt = attn_weights.size()
            loss_distill = self.kld_criterion(
                torch.log(attn_weights + 1e-6).reshape(b * L_vid, -1),
                torch.softmax(moment2txt_similarity, dim=-1).clone().detach().reshape(b * L_vid, -1)).mean(1) * moment_mask.reshape(-1)
            loss_distill = loss_distill.sum() / moment_mask.sum()

        else:
            loss_distill = 0.
        return {"loss_distill": loss_distill}

    def loss_orthogonal_dummy(self, outputs, targets, indices, log=True):
        dummy_tokens = outputs["dummy_tokens"]  # (n_dum, dim)
        if dummy_tokens.size(1) != 1:
            dummy_tokens_norm = dummy_tokens / dummy_tokens.norm(dim=2)[:, :, None]
            dummy_tokens_sim = torch.matmul(dummy_tokens_norm, dummy_tokens_norm.permute(0, 2, 1).detach())
            for i in range(len(dummy_tokens_sim)):
                dummy_tokens_sim[i].fill_diagonal_(0)
            loss_dummy_ortho = dummy_tokens_sim.abs().mean()
        else:
            loss_dummy_ortho=0.
        global_tokens = outputs["global_rep_tokens"]

        global_tokens_norm = global_tokens / global_tokens.norm(dim=1)[:, None]
        global_tokens_sim = torch.matmul(global_tokens_norm, global_tokens_norm.permute(1, 0).detach())
        for i in range(len(global_tokens_sim)):
            global_tokens_sim.fill_diagonal_(0)
        loss_dummy_ortho += global_tokens_sim.abs().mean()
        return {"loss_orthogonal_dummy": loss_dummy_ortho}


    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "ms_align": self.loss_contrastive_moment_sentence,
            "distill": self.loss_moment2txt_sim_distill,
            "orthogonal_dummy":self.loss_orthogonal_dummy
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency", "ms_align", "distill", "orthogonal_dummy"]
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    if "ms_align" == loss:
                        continue
                    if "distill" == loss:
                        continue
                    if "orthogonal_dummy" == loss:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    if args.a_feat_dir is None:
        model = CGDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            args=args
        )
        # model = SAMMIMIC(
        #     transformer,
        #     position_embedding,
        #     txt_position_embedding,
        #     txt_dim=args.t_feat_dim,
        #     vid_dim=args.v_feat_dim,
        #     num_queries=args.num_queries,
        #     input_dropout=args.input_dropout,
        #     aux_loss=args.aux_loss,
        #     contrastive_align_loss=args.contrastive_align_loss,
        #     contrastive_hdim=args.contrastive_hdim,
        #     span_loss_type=args.span_loss_type,
        #     use_txt_pos=args.use_txt_pos,
        #     n_input_proj=args.n_input_proj,
        #     args=args
        # )
    else:
        model = CGDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            args=args
        )
        # model = SAMMIMIC(
        #     transformer,
        #     position_embedding,
        #     txt_position_embedding,
        #     txt_dim=args.t_feat_dim,
        #     vid_dim=args.v_feat_dim,
        #     num_queries=args.num_queries,
        #     input_dropout=args.input_dropout,
        #     aux_loss=args.aux_loss,
        #     contrastive_align_loss=args.contrastive_align_loss,
        #     contrastive_hdim=args.contrastive_hdim,
        #     span_loss_type=args.span_loss_type,
        #     use_txt_pos=args.use_txt_pos,
        #     n_input_proj=args.n_input_proj,
        #     args=args
        # )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency,
                   "loss_ms_align": args.lw_ms_align,
                   "loss_distill": args.lw_distill,
                   "loss_orthogonal_dummy":args.lw_distill}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    # losses = ['spans', 'labels', 'saliency', 'ms_align', 'distill', 'orthogonal_dummy']
    losses = ['spans', 'labels', 'saliency', 'orthogonal_dummy']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For highlight detection datasets
    use_matcher = not (args.dset_name in ['youtube_uni', 'tvsum'])
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher, args=args
    )
    criterion.to(device)
    return model, criterion
