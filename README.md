# PPO MuJoCo (PyTorch + Gymnasium)

## 安裝

pip install -r requirements.txt

## 訓練
python train.py --env_id Hopper-v5
## or
python train.py --env_id Walker2d-v5

## 評估
python evaluate.py --env_id Hopper-v5 --ckpt checkpoints/ppo_Hopper-v5_final.pt --episodes 5

```bash
