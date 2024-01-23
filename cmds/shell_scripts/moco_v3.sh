# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
  # general
  -e|--epochs) epochs="$2"; shift; shift ;;
  -g|--GPU_NUM) GPU_NUM="$2"; shift; shift ;;
  -s|--split) split="$2"; shift; shift ;;
  -p|--port) port="$2"; shift; shift ;;
  --MASTER) MASTER="$2"; shift; shift ;;
  --NODE_NUM) NODE_NUM="$2"; shift; shift ;;
  --RANK) RANK="$2"; shift; shift ;;
  --msr) msr="$2"; shift; shift ;;
  --skip_tune) skip_tune="$2"; shift; shift ;;
  --skip_pretrain) skip_pretrain="$2"; shift; shift ;;

  --workers) workers="$2"; shift; shift ;;
  --warm_up_epochs) warm_up_epochs="$2"; shift; shift ;;
  --data) data=("$2"); shift; shift ;;
  --arch) arch=("$2"); shift; shift ;;
  --batch_size) batch_size=("$2"); shift; shift ;;
  --lr) lr=("$2"); shift; shift ;;
  --tune_lr) tune_lr=("$2"); shift; shift ;;
  --tune_batch_size) tune_batch_size=("$2"); shift; shift ;;
  --few_shot_tune_lr) few_shot_tune_lr=("$2"); shift; shift ;;
  --few_shot_tune_batch_size) few_shot_tune_batch_size=("$2"); shift; shift ;;
  --temp) temp=("$2"); shift; shift ;;
  --save_dir) save_dir=("$2"); shift; shift ;;
  --ckpt_pretrain) ckpt_pretrain=("$2"); shift; shift ;;
  --ckpt_tune) ckpt_tune=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;

  # test option
  --resume) resume=("$2"); shift; shift ;;
  --evaluate_pretrain) evaluate_pretrain=("$2"); shift; shift ;;
  --evaluate_pretrain_representation) evaluate_pretrain_representation=("$2"); shift; shift ;;
  --evaluate_moe_gate_selection) evaluate_moe_gate_selection=("$2"); shift; shift ;;
  --evaluate_moe_gate_selection_fix_trans) evaluate_moe_gate_selection_fix_trans=("$2"); shift; shift ;;

  # moe options
  --moe_data_distributed) moe_data_distributed=("$2"); shift; shift ;;
  --moe_experts) moe_experts=("$2"); shift; shift ;;
  --moe_top_k) moe_top_k=("$2"); shift; shift ;;
  --moe_mlp_ratio) moe_mlp_ratio=("$2"); shift; shift ;;
  --moe_noisy_gate_loss_weight) moe_noisy_gate_loss_weight=("$2"); shift; shift ;;
  --moe_gate_arch) moe_gate_arch=("$2"); shift; shift ;;
  --moe_contrastive_w) moe_contrastive_w=("$2"); shift; shift ;;
  --moe_contrastive_gate_proj_layers) moe_contrastive_gate_proj_layers=("$2"); shift; shift ;;
  --vmoe_noisy_std) vmoe_noisy_std=("$2"); shift; shift ;;
  --moe_same_for_all) moe_same_for_all=("$2"); shift; shift ;;
  --moe_gate_type) moe_gate_type=("$2"); shift; shift ;;
  # moe wassertein loss
  --moe_wassertein_gate) moe_wassertein_gate=("$2"); shift; shift ;;
  --moe_wassertein_gate_steps) moe_wassertein_gate_steps=("$2"); shift; shift ;;
  --moe_wassertein_gate_lr) moe_wassertein_gate_lr=("$2"); shift; shift ;;
  --moe_wassertein_neg_w) moe_wassertein_neg_w=("$2"); shift; shift ;;
  --moe_wassertein_gate_layers) moe_wassertein_gate_layers=("$2"); shift; shift ;;
  --moe_wassertein_gate_gp_w) moe_wassertein_gate_gp_w=("$2"); shift; shift ;;
  --moe_wassertein_gate_no_cls) moe_wassertein_gate_no_cls=("$2"); shift; shift ;;
  --moe_wassertein_gate_no_cls_w) moe_wassertein_gate_no_cls_w=("$2"); shift; shift ;;
  # moe esvit loss for gate
  --moe_esvit_gate) moe_esvit_gate=("$2"); shift; shift ;;
  --moe_esvit_gate_w) moe_esvit_gate_w=("$2"); shift; shift ;;
  # moe cls token gate
  --moe_cls_token_gate) moe_cls_token_gate=("$2"); shift; shift ;;
  # moe experts lr
  --moe_experts_lr_ratio) moe_experts_lr_ratio=("$2"); shift; shift ;;

  # iou gate
  --moe_iou_gate) moe_iou_gate=("$2"); shift; shift ;;
  --moe_iou_gate_alpha) moe_iou_gate_alpha=("$2"); shift; shift ;;
  --moe_iou_gate_threshold) moe_iou_gate_threshold=("$2"); shift; shift ;;
  --moe_iou_gate_similarity_mode) moe_iou_gate_similarity_mode=("$2"); shift; shift ;;
  --moe_gate_return_decoupled_activation) moe_gate_return_decoupled_activation=("$2"); shift; shift ;;
  --moe_gate_return_gated_activation) moe_gate_return_gated_activation=("$2"); shift; shift ;;

  # the local crops number
  --local_crops_number) local_crops_number=("$2"); shift; shift ;;

  # sim_rank
  --sim_rank_alpha) sim_rank_alpha=("$2"); shift; shift ;;
  *) echo "${1} is not found"; exit 125;
esac
done

epochs=${epochs:-200}
GPU_NUM=${GPU_NUM:-1}
split=${split:-full}
port=${port:-4833}
NODE_NUM=${NODE_NUM:-1}
msr=${msr:-"False"}
skip_tune=${skip_tune:-"False"}
skip_pretrain=${skip_pretrain:-"False"}

workers=${workers:-3}
warm_up_epochs=${warm_up_epochs:-40}
data=${data:-'placeholder'}
arch=${arch:-'vit_small'}
batch_size=${batch_size:-1024}
lr=${lr:-"1.5e-4"}
temp=${temp:-0.2}
save_dir=${save_dir:-"."}
ckpt_pretrain=${ckpt_pretrain:-"checkpoints"}
ckpt_tune=${ckpt_tune:-"checkpoints_tune"}
dataset=${dataset:-"imagenet100"}

resume=${resume:-""}
evaluate_pretrain=${evaluate_pretrain:-""}
evaluate_pretrain_representation=${evaluate_pretrain_representation:-""}
evaluate_moe_gate_selection=${evaluate_moe_gate_selection:-"False"}
evaluate_moe_gate_selection_fix_trans=${evaluate_moe_gate_selection_fix_trans:-"False"}

moe_data_distributed=${moe_data_distributed:-"False"}
moe_experts=${moe_experts:-"-1"}
moe_top_k=${moe_top_k:-2}
moe_mlp_ratio=${moe_mlp_ratio:-"-1"}
moe_noisy_gate_loss_weight=${moe_noisy_gate_loss_weight:-"0.01"}
moe_gate_arch=${moe_gate_arch:-""}
moe_contrastive_w=${moe_contrastive_w:-"-1"}
moe_contrastive_gate_proj_layers=${moe_contrastive_gate_proj_layers:-"-1"}
moe_gate_type=${moe_gate_type:-"noisy"}
vmoe_noisy_std=${vmoe_noisy_std:-"0"}
moe_same_for_all=${moe_same_for_all:-"False"}
# moe_wassertein
moe_wassertein_gate=${moe_wassertein_gate:-"False"}
moe_wassertein_gate_steps=${moe_wassertein_gate_steps:-"50"}
moe_wassertein_gate_lr=${moe_wassertein_gate_lr:-"3e-4"}
moe_wassertein_neg_w=${moe_wassertein_neg_w:-"0"}
moe_wassertein_gate_layers=${moe_wassertein_gate_layers:-"2"}
moe_wassertein_gate_gp_w=${moe_wassertein_gate_gp_w:-"1000"}
moe_wassertein_gate_no_cls=${moe_wassertein_gate_no_cls:-"False"}
moe_wassertein_gate_no_cls_w=${moe_wassertein_gate_no_cls_w:-"1.0"}
moe_esvit_gate_w=${moe_esvit_gate_w:-"1.0"}
# moe_wassertein
moe_esvit_gate=${moe_esvit_gate:-"False"}

moe_cls_token_gate=${moe_cls_token_gate:-"False"}

moe_experts_lr_ratio=${moe_experts_lr_ratio:-"1"}

moe_iou_gate=${moe_iou_gate:-"False"}
moe_iou_gate_alpha=${moe_iou_gate_alpha:-"0.5"}
moe_iou_gate_threshold=${moe_iou_gate_threshold:-"0.2"}
moe_iou_gate_similarity_mode=${moe_iou_gate_similarity_mode:-"False"}
moe_gate_return_decoupled_activation=${moe_gate_return_decoupled_activation:-"False"}
moe_gate_return_gated_activation=${moe_gate_return_gated_activation:-"False"}

local_crops_number=${local_crops_number:-"0"}

sim_rank_alpha=${sim_rank_alpha:-"-1"}

if [[ ${dataset} == "imagenet" ]]
then
  tune_lr=${tune_lr:-"3.0"}
  tune_batch_size=${tune_batch_size:-"4096"}
  few_shot_tune_batch_size=${few_shot_tune_batch_size:-"256"}
  few_shot_tune_epochs=${few_shot_tune_epochs:-"800"}
else
  tune_lr=${tune_lr:-"1.0"}
  tune_batch_size=${tune_batch_size:-"1024"}
  few_shot_tune_batch_size=${few_shot_tune_batch_size:-"256"}
  few_shot_tune_epochs=${few_shot_tune_epochs:-"800"}
fi

exp_name="${arch}_${dataset}_lr${lr}B${batch_size}E${epochs}"

if [[ ${moe_data_distributed} == "True" ]]
then
  exp_name="${exp_name}_dataDist"
fi

if [[ ${moe_experts} != "-1" ]]
then
  exp_name="${exp_name}_moeEpts${moe_experts}T${moe_top_k}"
  if [[ ${moe_mlp_ratio} != "-1" ]]
  then
    exp_name="${exp_name}Ratio${moe_mlp_ratio}"
  fi
  if [[ ${moe_gate_type} == "noisy_vmoe" ]]
  then
    exp_name="${exp_name}Gvmoe"
  fi
  if [[ ${moe_same_for_all} == "True" ]]
  then
    exp_name="${exp_name}SameForAll"
  fi
  if [[ ${moe_noisy_gate_loss_weight} != "0.01" ]]
  then
    exp_name="${exp_name}Ngw${moe_noisy_gate_loss_weight}"
  fi
  if [[ ${moe_contrastive_w} != "-1" ]]
  then
    exp_name="${exp_name}CLw${moe_contrastive_w}"
    if [[ ${moe_contrastive_gate_proj_layers} != "-1" ]]
    then
      exp_name="${exp_name}l${moe_contrastive_gate_proj_layers}"
    fi
    if [[ ${moe_wassertein_gate} == "True" ]]
    then
      exp_name="${exp_name}WstLr${moe_wassertein_gate_lr}S${moe_wassertein_gate_steps}"
      if [[ ${moe_wassertein_gate_layers} != "2" ]]
      then
        exp_name="${exp_name}L${moe_wassertein_gate_layers}"
      fi
      if [[ ${moe_wassertein_gate_gp_w} != "1000" ]]
      then
        exp_name="${exp_name}gpW${moe_wassertein_gate_gp_w}"
      fi
      if [[ ${moe_wassertein_gate_no_cls} == "True" ]]
      then
        exp_name="${exp_name}NoCls"
        if [[ ${moe_wassertein_gate_no_cls_w} != "1.0" ]]
        then
          exp_name="${exp_name}NoClsRestW${moe_wassertein_gate_no_cls_w}"
        fi
      fi
      exp_name="${exp_name}Negw${moe_wassertein_neg_w}"
    fi
    if [[ ${moe_esvit_gate} == "True" ]]
    then
      exp_name="${exp_name}EsG"
      if [[ ${moe_esvit_gate_w} != "1.0" ]]
      then
        exp_name="${exp_name}W${moe_esvit_gate_w}"
      fi
    fi
    if [[ ${moe_iou_gate} == "True" ]]
    then
      exp_name="${exp_name}IoUThre${moe_iou_gate_threshold}Alpha${moe_iou_gate_alpha}"
    fi
    if [[ ${moe_gate_return_gated_activation} == "True" ]]
    then
      exp_name="${exp_name}CLgate"
    fi
    if [[ ${moe_iou_gate_similarity_mode} == "True" ]]
    then
      exp_name="${exp_name}Sim"
    fi
    if [[ ${moe_cls_token_gate} == "True" ]]
    then
      exp_name="${exp_name}ClsG"
    fi
  fi
  if [[ ${moe_gate_arch} != "" ]]
  then
    exp_name="${exp_name}_${moe_gate_arch}"
  fi
  if [[ ${moe_gate_return_decoupled_activation} == "True" ]]
  then
    exp_name="${exp_name}_decoupleGate"
  fi
  if [[ ${moe_experts_lr_ratio} != "1" ]]
    then
      exp_name="${exp_name}_moeLrR${moe_experts_lr_ratio}"
  fi
fi

if [[ ${local_crops_number} != "0" ]]
then
  exp_name="${exp_name}_localCropsN${local_crops_number}"
fi

if [[ ${sim_rank_alpha} != "-1" ]]
then
  exp_name="${exp_name}_simRankW${sim_rank_alpha}"
fi

if [[ ${evaluate_moe_gate_selection} == "True" ]]
then
  exp_name="${exp_name}_evalMoE"
  if [[ ${evaluate_moe_gate_selection_fix_trans} == "True" ]]
  then
    exp_name="${exp_name}_FixTrans"
  fi
fi

if [[ ${NODE_NUM} != "1" ]]
then
  launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} \
  --nnodes=${NODE_NUM} --node_rank=${RANK} --master_addr=${MASTER}"
elif [[ ${msr} == "True" ]]
then
  launch_cmd="python"
else
  launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"
fi

cmd="${launch_cmd} main_moco.py ${exp_name} \
  -a ${arch} -b ${batch_size} \
  --optimizer=adamw --lr=${lr} --weight-decay=.1 \
  --epochs=${epochs} --warmup-epochs=${warm_up_epochs} \
  --stop-grad-conv1 --moco-m-cos --moco-t=${temp} \
  --dist-url tcp://localhost:${port} \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --dataset ${dataset} --workers ${workers} \
  --save_dir ${save_dir}/${ckpt_pretrain}"


############## linear evaluation cmd ##############
fine_tune_cmd="bash cmds/shell_scripts/tunePretrained_sweep.sh \
--pretrain_name ${exp_name} --dataset ${dataset} --batch_size ${tune_batch_size} \
--pretrain ${save_dir}/${ckpt_pretrain}/${exp_name}/checkpoint_final.pth.tar \
--save_dir ${save_dir}/${ckpt_tune} --arch ${arch} -g ${GPU_NUM} -p ${port}"

if [[ ${moe_data_distributed} == "True" ]]
then
  cmd="${cmd} --moe-data-distributed"
  fine_tune_cmd="${fine_tune_cmd} --moe_data_distributed True"
fi

if [[ ${moe_experts} != "-1" ]]
then
  cmd="${cmd} --moe-experts ${moe_experts} --moe-top-k ${moe_top_k} --moe-noisy-gate-loss-weight ${moe_noisy_gate_loss_weight} \
  --moe-mlp-ratio ${moe_mlp_ratio} --vmoe-noisy-std ${vmoe_noisy_std} --moe-gate-type ${moe_gate_type} --data ${data}"
  fine_tune_cmd="${fine_tune_cmd} --moe_experts ${moe_experts} --moe_top_k ${moe_top_k} --moe_noisy_gate_loss_weight ${moe_noisy_gate_loss_weight} \
  --moe_mlp_ratio ${moe_mlp_ratio} --vmoe_noisy_std ${vmoe_noisy_std} --moe_gate_type ${moe_gate_type} --data ${data}"
  if [[ ${moe_contrastive_w} != "-1" ]]
  then
    cmd="${cmd} --moe-contrastive-weight ${moe_contrastive_w}"
    if [[ ${moe_contrastive_gate_proj_layers} != "-1" ]]
    then
      cmd="${cmd} --moe-contrastive-gate-proj-layers ${moe_contrastive_gate_proj_layers}"
    fi
    if [[ ${moe_wassertein_gate} == "True" ]]
    then
      cmd="${cmd} --moe-contrastive-gate-proj-layers ${moe_contrastive_gate_proj_layers} \
                  --moe-wassertein-gate \
                  --moe-wassertein-gate-steps ${moe_wassertein_gate_steps} \
                  --moe-wassertein-gate-lr ${moe_wassertein_gate_lr} \
                  --moe-wassertein-neg-w ${moe_wassertein_neg_w} \
                  --moe-wassertein-gate-layers ${moe_wassertein_gate_layers} \
                  --moe-wassertein-gate-gp-w ${moe_wassertein_gate_gp_w}"
    fi
    if [[ ${moe_wassertein_gate_no_cls} == "True" ]]
    then
      cmd="${cmd} --moe-wassertein-gate-no-cls --moe-wassertein-gate-no-cls-w ${moe_wassertein_gate_no_cls_w}"
    fi
    if [[ ${moe_esvit_gate} == "True" ]]
    then
      cmd="${cmd} --moe-esvit-gate --moe-wassertein-gate-no-cls-w ${moe_esvit_gate_w}"
    fi
    if [[ ${moe_iou_gate} == "True" ]]
    then
      cmd="${cmd} --moe-iou-gate --moe-iou-gate-threshold ${moe_iou_gate_threshold} \
      --moe-iou-gate-alpha ${moe_iou_gate_alpha}"
    fi
    if [[ ${moe_iou_gate_similarity_mode} == "True" ]]
    then
      cmd="${cmd} --moe-iou-gate-similarity-mode"
    fi
    if [[ ${moe_cls_token_gate} == "True" ]]
    then
      cmd="${cmd} --moe-cls-token-gate"
    fi
  fi
  if [[ ${moe_gate_arch} != "" ]]
  then
    cmd="${cmd} --moe-gate-arch ${moe_gate_arch}"
    fine_tune_cmd="${fine_tune_cmd} --moe_gate_arch ${moe_gate_arch}"
  fi
  if [[ ${moe_gate_return_decoupled_activation} == "True" ]]
  then
    cmd="${cmd} --moe-gate-return-decoupled-activation"
  fi
  if [[ ${moe_gate_return_gated_activation} == "True" ]]
  then
    cmd="${cmd} --moe-gate-return-gated-activation"
  fi
  if [[ ${moe_experts_lr_ratio} != "1" ]]
  then
    cmd="${cmd} --experts_lr_ratio ${moe_experts_lr_ratio}"
  fi
  if [[ ${moe_same_for_all} == "True" ]]
  then
    cmd="${cmd} --moe-same-for-all"
    fine_tune_cmd="${fine_tune_cmd} --moe_same_for_all ${moe_same_for_all}"
  fi
fi

if [[ ${local_crops_number} != "0" ]]
then
  cmd="${cmd} --local_crops_number ${local_crops_number}"
fi

if [[ ${sim_rank_alpha} != "-1" ]]
then
  cmd="${cmd} --sim_rank_alpha ${sim_rank_alpha} --sim_rank"
fi

if [[ ${resume} != "" ]]
then
  cmd="${cmd} --resume ${resume}"
fi

if [[ ${evaluate_pretrain} != "" ]]
then
  cmd="${cmd} --evaluate_pretrain"
  if [[ ${evaluate_pretrain_representation} != "" ]]
  then
    cmd="${cmd} --evaluate_pretrain_representation"
  fi
fi

if [[ ${evaluate_moe_gate_selection} == "True" ]]
then
  cmd="${cmd} --evaluate_moe_gate_selection"
  if [[ ${evaluate_moe_gate_selection_fix_trans} == "True" ]]
  then
    cmd="${cmd} --evaluate_moe_gate_selection_fix_trans"
  fi
fi

################### few shot cmd ###################
fewshot_cmd="${fine_tune_cmd} --customSplit imagenet_1percent --customSplitName 1perc \
 --batch_size ${few_shot_tune_batch_size} --epochs ${few_shot_tune_epochs}"

###################################################
if [[ ${skip_pretrain} != "True" ]]
then
echo ${cmd}
${cmd}
fi

if [[ ${skip_tune} != "True" ]] && [[ ${msr} != "True" ]]
then
  echo ${fine_tune_cmd}
  ${fine_tune_cmd}

  echo ${fewshot_cmd}
  ${fewshot_cmd}
fi