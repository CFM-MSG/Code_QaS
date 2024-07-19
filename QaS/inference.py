import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from qas.config import TestOptions
from qas.model import build_model
from qas.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from standalone_eval.eval import eval_submission
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data,
            verbose=opt.debug, match_number=not opt.debug
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


# for HL
@torch.no_grad()
def compute_hl_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []

    topk = 5 # top-5 map

    video_ap_collected = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        outputs = model(**model_inputs)

        # loss meters
        if criterion:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))


        preds = outputs['saliency_scores']

        for meta, pred in zip(query_meta, preds):
            pred = pred
            label = meta['label'] # raw label

            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"
            for i in range(20):
                cur_pred = pred[:len(label)]
                inds = torch.argsort(cur_pred, descending=True, dim=-1)

                # video_id = self.get_video_id(idx)
                cur_label = torch.Tensor(label)[:, i]
                cur_label = torch.where(cur_label > cur_label.median(), 1.0, .0)

                cur_label = cur_label[inds].tolist()[:topk]

                # if (num_gt := sum(cur_label)) == 0:
                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc

                video_ap.append(ap)
        video_ap_collected.append(video_ap)  

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))
    

    # tensorboard writer
    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return submmission, loss_meters 



@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        outputs = model(epoch=epoch_i, **model_inputs)
        # inference
        bsz = len(query_meta)
        num_props = outputs['width'].shape[0] // bsz
        pos_vid_concept = outputs['pos_vid_concepts']
        txt_concepts = outputs['txt_concepts']

        num_concept = pos_vid_concept.shape[1]
        proposal = F.normalize(pos_vid_concept,dim=-1, p=2).reshape(bsz*num_props, num_concept, -1)
        txt_conc = F.normalize(txt_concepts,dim=-1, p=2).unsqueeze(1).expand(bsz, num_props, num_concept, -1).reshape(bsz*num_props, num_concept, -1)
        pos_samilarity = torch.diagonal(torch.bmm(proposal, txt_conc.transpose(1, 2)), dim1=-2, dim2=-1).sum(dim=-1)
        pos_samilarity = pos_samilarity.reshape(bsz, num_props)
        idx = pos_samilarity.argsort(dim=-1, descending=True)
        pos_samilarity_ranked = pos_samilarity.sort(dim=-1, descending=True)[0]


        width = outputs['width'].view(bsz, num_props).gather(index=idx, dim=-1) # bsz, num_props
        center = outputs['center'].view(bsz, num_props).gather(index=idx, dim=-1)
       
        pred_spans = torch.stack([torch.clamp(center-width/2, min=0), 
                                  torch.clamp(center+width/2, max=1)], dim=-1) * query_meta[0]["duration"] # bsz, num_props, 2
        
        # saliency
        _saliency_scores = outputs['saliency_score'].half()
        saliency_scores = []
        valid_vid_mask = outputs['video_mask'].sum(dim=-1).cpu().tolist()
        for j in range(len(valid_vid_mask)):
            saliency_scores.append(_saliency_scores[j, : int(valid_vid_mask[j])].tolist())
        scores = pos_samilarity.sort(dim=-1, descending=True)[0]
        # compose predictions
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            # # (#queries, 3), list: [0:[st(float), ed(float), score(float)], 1:[]...]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1)
            cur_ranked_preds = torch.round(cur_ranked_preds).tolist()
            # if not opt.no_sort_results: # rank vmr result based on score or not
            #     cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx],
            )
            mr_res.append(cur_query_pred)


        if opt.debug:
            break

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return mr_res, loss_meters


def get_eval_res(model, eval_loader, opt, epoch_i, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, tb_writer)  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )
    
    # tvsum 
    if opt.dset_name in ['tvsum']:
        metrics, eval_loss_meters = compute_hl_results(model, eval_loader, opt, epoch_i, tb_writer)
        
        # to match original save format
        submission = [
            {"brief": metrics}
        ]
        submission_path = os.path.join(opt.results_dir, "latest_metric.jsonl")
        save_jsonl(submission, submission_path)

        return submission[0], submission[0], eval_loss_meters, [submission_path]

    else:
        submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, tb_writer)
            
        if opt.no_sort_results:
            save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")

        metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
            submission, opt, eval_dataset.data, save_submission_filename)
        return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)


    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, optimizer, lr_scheduler


def start_inference(train_opt=None, split=None, splitfile=None):
    if train_opt is not None:
        opt = TestOptions().parse(train_opt.a_feat_dir)
    else:
        opt = TestOptions().parse()
    if split is not None:
        opt.eval_split_name = split
    if splitfile is not None:
        opt.eval_path = splitfile


    logger.info("Setup config, data and model...")


    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    if opt.eval_split_name == 'val':
        loadlabel = True
    else:
        loadlabel = False
    print("Video Evaluation")
    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        vocab_path=opt.vocab_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        vocab_size=opt.vocab_size,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=loadlabel,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
    )


    model, _, _ = setup_model(opt)

    save_submission_filename = "hl_{}_submission.jsonl".format(
        opt.eval_split_name)
    # save_submission_filename = "inference_{}_{}_{}_preds.jsonl".format(
    #     opt.dset_name, opt.eval_split_name, opt.eval_id)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename)
    if opt.eval_split_name == 'val':
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

from sys import argv
if __name__ == '__main__':
    _,_,_,_,split,_,splitfile = argv

    start_inference(split=split, splitfile=splitfile)
