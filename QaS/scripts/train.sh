dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=glove
results_root=results
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
vocab_path=data/glove.pkl
eval_split_name=val

######## setup video+text features
feat_root=.../features


# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "glove" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=300
fi

#### training
bsz=32
max_q_l=32
vocab_size=6777
neg_type=lnb

# loss coef
pos_rec_coef=1
neg_rec_coef=1
ref_rec_coef=1
rank_loss_coef=1
div_loss_coef=0.5
sal_trip_coef=0.5



PYTHONPATH=$PYTHONPATH:. python qas/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--vocab_path ${vocab_path} \
--vocab_size ${vocab_size} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--max_q_l ${max_q_l} \
--neg_type ${neg_type} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--pos_rec_coef ${pos_rec_coef} \
--neg_rec_coef ${neg_rec_coef} \
--ref_rec_coef ${ref_rec_coef} \
--rank_loss_coef ${rank_loss_coef} \
--div_loss_coef ${div_loss_coef} \
--sal_trip_coef ${sal_trip_coef} \


${@:1}

