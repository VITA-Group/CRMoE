# pretrain for 100 epochs on imagenet (tested on 16 V100s, downstream tasks would be automatically evaluated after pretraining)
save_dir="."
PATH2IMAGENET="yourImagenetPath"

pretrain_lr=5e-4

moe_experts=16
moe_iou_gate_alpha=0.3
moe_iou_gate_threshold=0.2
moe_contrastive_w=1e-3

bash cmds/shell_scripts/moco_v3.sh -g 16 --arch moe_vit_small --save_dir ${save_dir} --data ${PATH2IMAGENET} \
--moe_experts ${moe_experts} --moe_top_k 2 --moe_mlp_ratio 2 --moe_noisy_gate_loss_weight 0.01 \
--dataset imagenet --batch_size 1024 --epochs 100 --lr ${pretrain_lr}  \
--moe_contrastive_w ${moe_contrastive_w} --moe_iou_gate True --moe_iou_gate_alpha ${moe_iou_gate_alpha} --moe_iou_gate_threshold ${moe_iou_gate_threshold} \
--moe_gate_type "noisy_vmoe" --moe_gate_return_gated_activation True --moe_data_distributed True --workers 3 -p 5566
