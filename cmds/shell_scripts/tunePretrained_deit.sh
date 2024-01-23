# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
  # general
  -e|--epochs) epochs="$2"; shift; shift ;;
  -p|--port) port="$2"; shift; shift ;;
  -w|--workers) workers="$2"; shift; shift ;;
  -g|--GPU_NUM) GPU_NUM="$2"; shift; shift ;;
  --batch_size) batch_size="$2"; shift; shift ;;
  --test_interval) test_interval="$2"; shift; shift ;;

  --data) data=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;
  --arch) arch=("$2"); shift; shift ;;
  --lr) lr=("$2"); shift; shift ;;
  --weight_decay) weight_decay=("$2"); shift; shift ;;
  --pretrain) pretrain=("$2"); shift; shift ;;
  --pretrain_name) pretrain_name=("$2"); shift; shift ;;
  --save_dir) save_dir=("$2"); shift; shift ;;
  --kaiming_sched) kaiming_sched=("$2"); shift; shift ;;

  --customSplit) customSplit=("$2"); shift; shift ;;
  --customSplitName) customSplitName=("$2"); shift; shift ;;

  # moe
  --moe_experts) moe_experts=("$2"); shift; shift ;;
  --moe_top_k) moe_top_k=("$2"); shift; shift ;;
  --moe_mlp_ratio) moe_mlp_ratio=("$2"); shift; shift ;;
  --moe_gate_arch) moe_gate_arch=("$2"); shift; shift ;;
  --moe_gate_type) moe_gate_type=("$2"); shift; shift ;;
  --vmoe_noisy_std) vmoe_noisy_std=("$2"); shift; shift ;;

  --moe_data_distributed) moe_data_distributed=("$2"); shift; shift ;;

  # moe-iou-gate
  --moe_contrastive_weight) moe_contrastive_weight=("$2"); shift; shift ;;
  --moe_contrastive_supervised ) moe_contrastive_supervised=("$2"); shift; shift ;;

  --tuneFromFirstFC ) tuneFromFirstFC=("$2"); shift; shift ;;
  --finetune_in_train) finetune_in_train=("$2"); shift; shift ;;

#  --moe_experts_prune) moe_experts_prune=("$2"); shift; shift ;;

  *) echo "${1} is not found"; exit 125;
esac
done

epochs=${epochs:-150}
port=${port:-4833}
workers=${workers:-4}
GPU_NUM=${GPU_NUM:-1}
batch_size=${batch_size:-1024}
test_interval=${test_interval:-1}

data=${data:-'placeholder'}
dataset=${dataset:-'imagenet'}
arch=${arch:-'vit_small'}
lr=${lr:-"5e-4"}
weight_decay=${weight_decay:-"0.05"}
pretrain=${pretrain:-"None"}
pretrain_name=${pretrain_name:-"None"}
save_dir=${save_dir:-"checkpoints"}
kaiming_sched=${kaiming_sched:-"False"}

customSplit=${customSplit:-""}
customSplitName=${customSplitName:-""}

# moe
moe_experts=${moe_experts:-"4"}
moe_top_k=${moe_top_k:-"2"}
moe_mlp_ratio=${moe_mlp_ratio:-"-1"}
moe_gate_arch=${moe_gate_arch:-""}
moe_gate_type=${moe_gate_type:-"noisy"}
vmoe_noisy_std=${vmoe_noisy_std:-"0"}
finetune_in_train=${finetune_in_train:-"False"}
tuneFromFirstFC=${tuneFromFirstFC:-"False"}
#moe_experts_prune=${moe_experts_prune:-"-1"}

moe_contrastive_weight=${moe_contrastive_weight:-"-1"}
moe_contrastive_supervised=${moe_contrastive_supervised:-"False"}

moe_data_distributed=${moe_data_distributed:-"False"}

exp="Deit_lr${lr}B${batch_size}E${epochs}_${dataset}"

if [[ ${moe_contrastive_weight} != "-1" ]]
then
  exp=${exp}_gateReg${moe_contrastive_weight}
  if [[ ${moe_contrastive_supervised} == "True" ]]
  then
    exp=${exp}Sup
  fi
fi

if [[ ${weight_decay} != "0.05" ]]
then
  exp=${exp}_wd${weight_decay}
fi

if [[ ${customSplit} != "" ]]
then
  exp=${exp}_${customSplitName}
fi

if [[ ${kaiming_sched} == "True" ]]
then
  exp=${exp}_KMsched
fi

if [[ ${finetune_in_train} == "True" ]]
then
  exp=${exp}_FTtrain
fi

if [[ ${tuneFromFirstFC} == "True" ]]
then
  exp=${exp}_TuneFromFirstFc
fi

#if [[ ${moe_experts_prune} != "-1" ]]
#then
#  exp="${exp}_tuneMoEp${moe_experts_prune}"
#fi

if [[ ${pretrain} != "None" ]]
then
  exp=${exp}__${pretrain_name}
fi

launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"

cmd="${launch_cmd} main_finetune_deit.py ${exp} --save_dir ${save_dir} --arch ${arch} --lr ${lr} --epochs ${epochs} \
--batch-size ${batch_size} --weight-decay ${weight_decay} \
--data ${data} --dataset ${dataset} --num_workers ${workers} --moe-gate-type ${moe_gate_type} --vmoe-noisy-std ${vmoe_noisy_std} \
--moe-experts ${moe_experts} --moe-top-k ${moe_top_k} --moe-mlp-ratio ${moe_mlp_ratio} --test-interval ${test_interval} --moe-contrastive-weight ${moe_contrastive_weight}"

if [[ ${moe_contrastive_supervised} == "True" ]]
then
  cmd="${cmd} --moe-contrastive-supervised"
fi

if [[ ${pretrain} != "None" ]]
then
  cmd="${cmd} --pretrained ${pretrain}"
fi

if [[ ${customSplit} != "" ]]
then
  cmd="${cmd} --customSplit ${customSplit}"
fi

if [[ ${moe_data_distributed} == "True" ]]
then
  cmd="${cmd} --moe-data-distributed"
fi

if [[ ${kaiming_sched} == "True" ]]
then
  cmd="${cmd} --kaiming-sched"
fi

if [[ ${finetune_in_train} == "True" ]]
then
  cmd="${cmd} --finetune_in_train"
fi

if [[ ${tuneFromFirstFC} == "True" ]]
then
  cmd="${cmd} --tuneFromFirstFC"
fi

echo ${cmd}
${cmd}
