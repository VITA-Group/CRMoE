# crmoe

## Environment Setting
```shell
conda create -n crmoe python=3.6 -y
conda activate crmoe
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

# fastmoe v0.3.0
git clone https://github.com/laekov/fastmoe.git
cd fastmoe
git checkout 4edeccd95a66d1f2d5eb7ec7a9e40a9f67e0e393
psudo() { sudo env PATH="$PATH" "$@"; }
psudo python setup.py install
pip install dm-tree
pip install tensorboard tensorboardX
pip install timm==0.3.2
```

## Running Cmds
Available at [cmds/execute_hist](cmds/execute_hist).

## Acknowledge
Partial of this code comes from [MocoV3](https://github.com/facebookresearch/moco-v3), [FastMoE](https://github.com/laekov/fastmoe) and [Deit](https://github.com/facebookresearch/deit).