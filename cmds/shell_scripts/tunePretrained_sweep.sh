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
  --arch) arch=("$2"); shift; shift ;;
  --pretrain) pretrain=("$2"); shift; shift ;;
  --pretrain_name) pretrain_name=("$2"); shift; shift ;;
  --save_dir) save_dir=("$2"); shift; shift ;;

  # finetune
  --fine_tune) fine_tune=("$2"); shift; shift ;;
  --customSplit) customSplit=("$2"); shift; shift ;;
  --customSplitName) customSplitName=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;

  # moe
  --moe_experts) moe_experts=("$2"); shift; shift ;;
  --moe_top_k) moe_top_k=("$2"); shift; shift ;;
  --moe_mlp_ratio) moe_mlp_ratio=("$2"); shift; shift ;;
  --moe_noisy_gate_loss_weight) moe_noisy_gate_loss_weight=("$2"); shift; shift ;;
  --moe_data_distributed) moe_data_distributed=("$2"); shift; shift ;;
  --moe_gate_arch) moe_gate_arch=("$2"); shift; shift ;;
  --moe_experts_prune) moe_experts_prune=("$2"); shift; shift ;;
  --moe_noisy_train) moe_noisy_train=("$2"); shift; shift ;;
  --moe_gate_type) moe_gate_type=("$2"); shift; shift ;;
  --vmoe_noisy_std) vmoe_noisy_std=("$2"); shift; shift ;;
  --moe_same_for_all) moe_same_for_all=("$2"); shift; shift ;;

  --sep_path) sep_path=("$2"); shift; shift ;;

  # distill
  --skip_tune) skip_tune=("$2"); shift; shift ;;
  --distill) distill=("$2"); shift; shift ;;
  --distill_lr) distill_lr=("$2"); shift; shift ;;
  --distill_temp) distill_temp=("$2"); shift; shift ;;
  --distill_batch_size) distill_batch_size=("$2"); shift; shift ;;
  --distill_epochs) distill_epochs=("$2"); shift; shift ;;

  --speed_test) speed_test=("$2"); shift; shift ;;

  --resume) resume=("$2"); shift; shift ;;

  *) echo "${1} is not found"; exit 125;
esac
done

epochs=${epochs:-90}
port=${port:-4833}
workers=${workers:-3}
GPU_NUM=${GPU_NUM:-1}
batch_size=${batch_size:-1024}
test_interval=${test_interval:-1}

data=${data:-'placeholder'}
arch=${arch:-'vit_small'}
pretrain=${pretrain:-"None"}
pretrain_name=${pretrain_name:-"None"}
save_dir=${save_dir:-"checkpoints"}


fine_tune=${fine_tune:-"False"}
customSplit=${customSplit:-""}
customSplitName=${customSplitName:-""}
dataset=${dataset:-"imagenet100"}

# moe
moe_experts=${moe_experts:-"4"}
moe_top_k=${moe_top_k:-"2"}
moe_mlp_ratio=${moe_mlp_ratio:-"-1"}
moe_noisy_gate_loss_weight=${moe_noisy_gate_loss_weight:-"0.01"}
moe_data_distributed=${moe_data_distributed:-"False"}
moe_gate_arch=${moe_gate_arch:-""}
moe_experts_prune=${moe_experts_prune:-"-1"}
moe_noisy_train=${moe_noisy_train:-"False"}
moe_gate_type=${moe_gate_type:-"noisy"}
vmoe_noisy_std=${vmoe_noisy_std:-"0"}
moe_same_for_all=${moe_same_for_all:-"False"}

sep_path=${sep_path:-"0"}

skip_tune=${skip_tune:-"False"}
distill=${distill:-"False"}
distill_temp=${distill_temp:-0.1}
distill_batch_size=${distill_batch_size:-2048}
distill_epochs=${distill_epochs:-"100"}

speed_test=${speed_test:-"False"}

resume=${resume:-"None"}

exp="linearSweep_B${batch_size}E${epochs}_${dataset}"

if [[ ${customSplit} != "" ]]
then
  exp=${exp}_${customSplitName}
fi

if [[ ${fine_tune} == "True" ]]
then
  exp=Tune_${exp}
fi

if [[ ${moe_experts_prune} != "-1" ]]
then
  exp="${exp}_tuneMoEp${moe_experts_prune}"
fi

if [[ ${moe_noisy_train} == "True" ]]
then
  exp="${exp}_NoiseT"
fi

if [[ ${sep_path} != "0" ]]
then
  exp="${exp}_sepPath${sep_path}"
fi

if [[ ${speed_test} == "True" ]]
then
  exp=${exp}_testSpeed
fi

if [[ ${pretrain} != "None" ]]
then
  exp=${exp}__${pretrain_name}
fi

if [[ ${distill} == "True" ]]
then
  distill_name="${exp}_dtlT${distill_temp}Lr${distill_lr}B${distill_batch_size}E${distill_epochs}"
fi

launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"

cmd="${launch_cmd} main_lincls_sweep.py ${exp} --save_dir ${save_dir} -a ${arch} --epochs ${epochs} \
--multiprocessing-distributed --world-size 1 --rank 0 --batch-size ${batch_size} \
--data ${data} --dataset ${dataset} --workers ${workers} \
--moe-experts ${moe_experts} --moe-top-k ${moe_top_k} --moe-mlp-ratio ${moe_mlp_ratio} --moe-noisy-gate-loss-weight=${moe_noisy_gate_loss_weight} \
--moe-gate-type ${moe_gate_type} --vmoe-noisy-std ${vmoe_noisy_std} \
 --test-interval ${test_interval}"

if [[ ${pretrain} != "None" ]]
then
  cmd="${cmd} --pretrained ${pretrain}"
fi

if [[ ${resume} != "None" ]]
then
  cmd="${cmd} --resume ${resume}"
fi

if [[ ${moe_same_for_all} == "True" ]]
then
  cmd="${cmd} --moe-same-for-all"
fi

if [[ ${customSplit} != "" ]]
then
  cmd="${cmd} --customSplit ${customSplit}"
fi

if [[ ${fine_tune} == "True" ]]
then
  cmd="${cmd} --fine-tune"
fi

if [[ ${moe_data_distributed} == "True" ]]
then
  cmd="${cmd} --moe-data-distributed"
fi

if [[ ${moe_gate_arch} != "" ]]
then
  cmd="${cmd} --moe-gate-arch ${moe_gate_arch}"
fi

if [[ ${moe_experts_prune} != "-1" ]]
then
  cmd="${cmd} --moe-experts-prune ${moe_experts_prune}"
fi

if [[ ${moe_noisy_train} == "True" ]]
then
  cmd="${cmd} --moe-noisy-train"
fi

if [[ ${sep_path} != "0" ]]
then
  cmd="${cmd} --sep-path ${sep_path}"
fi

if [[ ${speed_test} == "True" ]]
then
  cmd="${cmd} --speed_test"
fi

cmd_distill="${launch_cmd} main_lincls_sweep.py ${distill_name} --save_dir ${save_dir} -a ${arch} --lr ${distill_lr} --epochs ${distill_epochs} \
--multiprocessing-distributed --world-size 1 --rank 0 --batch-size ${distill_batch_size} \
--data ${data} --dataset ${dataset} --workers ${workers} --wd 1e-4 \
--moe-experts ${moe_experts} --moe-top-k ${moe_top_k} --moe-noisy-gate-loss-weight=${moe_noisy_gate_loss_weight} \
--test-interval 1 --fine-tune --pretrained ${pretrain} \
--distillation --distillation_temp ${distill_temp} --distillation_checkpoint ${save_dir}/${exp}/model_best.pth.tar"

if [[ ${skip_tune} != "True" ]]
then
  echo ${cmd}
  ${cmd}
fi

if [[ ${distill} == "True" ]]
then
  echo ${cmd_distill}
  ${cmd_distill}
fi
