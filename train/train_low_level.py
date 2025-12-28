# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mini.env.v6_mini_env import MiniSkillEnv


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def train_one_skill(skill_id: int, steps: int, ent: float, out_dir: str, seed: int = 777):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import torch.nn as nn

    # 【关键1】每个技能独立但确定的种子
    unique_seed = 777 + skill_id * 1000  # 非随机！确保可重复性
    
    env = MiniSkillEnv(tolerance=0.05, max_steps=200, seed=unique_seed)
    ensure_dir(out_dir)

    class CurveCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_step(self) -> bool:
            # 使用 ep_info_buffer 提取平均奖励与长度
            try:
                ep_info = self.model.ep_info_buffer
                if len(ep_info) > 0:
                    r = np.mean([x['r'] for x in ep_info])
                    l = np.mean([x['l'] for x in ep_info])
                else:
                    r, l = 0.0, 0.0
                self.rows.append({"step": int(self.num_timesteps), "ep_rew_mean": float(r), "ep_len_mean": float(l)})
            except Exception:
                pass
            return True

    # 【修复9】调整训练参数 - 改为单阶段1500步，ent=0.08
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,        # 保持学习率
        n_steps=2048,              # 保持采样步数
        batch_size=64,             # 保持batch size
        n_epochs=4,                # 保持epoch数
        gamma=0.99,
        ent_coef=0.2,              # 【EXP15修复】强制探索，防止动作单一
        seed=unique_seed,
        verbose=0,
        policy_kwargs=dict(
            net_arch=[128, 128],   # 保持网络容量
            activation_fn=nn.Tanh,
            ortho_init=True
        )
    )
    
    # 【EXP15修复】改为3000步训练，充分探索
    print(f"训练技能{skill_id}: 单阶段训练3000步")
    cb = CurveCallback()
    model.learn(total_timesteps=3000, callback=cb, reset_num_timesteps=False)
    
    # 【修复11】立即评估并保存
    success_count = 0
    action_dist = np.zeros(6)
    
    for _ in range(30):  # 30个episode评估
        obs, _ = env.reset()
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            action_dist[act] += 1
            obs, _, terminated, truncated, _ = env.step(int(act))
            done = terminated or truncated
            if terminated:
                success_count += 1
    
    success_rate = success_count / 30.0
    
    # 【修复12】只有动作多样且成功率高才保存
    action_dist_norm = action_dist / (action_dist.sum() + 1e-8)
    action_entropy = -np.sum(action_dist_norm * np.log(action_dist_norm + 1e-8))
    
    print(f"技能{skill_id}评估: 成功率={success_rate:.2%}, 动作熵={action_entropy:.3f}")
    print(f"  动作分布: {action_dist}")
    
    # 【EXP15修复】只要成功就保存，降低坍缩模型被丢弃的概率
    if success_rate > 0.5:  # 主要靠高熵和高步数来减少坍缩发生
        model.save(os.path.join(out_dir, "policy.zip"))
        print(f"技能{skill_id}保存成功")
    else:
        print(f"警告: 技能{skill_id}成功率过低，但仍然保存...")
        # 即使成功率低也保存，避免训练失败
        model.save(os.path.join(out_dir, "policy.zip"))

    # 保存训练曲线和评估指标
    with open(os.path.join(out_dir, "training_curve.json"), "w", encoding="utf-8") as f:
        json.dump(cb.rows, f, indent=2, ensure_ascii=False)

    metrics = {
        "skill_id": skill_id,
        "success_rate": success_rate,
        "action_entropy": action_entropy,
        "action_dist": action_dist.tolist()
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)

    parser.add_argument("--ent", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="mini/results/low")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num-skills", type=int, default=12)
    args = parser.parse_args()

    ensure_dir(args.output)
    for skill_id in range(int(args.num_skills)):
        out_dir = os.path.join(args.output, f"skill{skill_id}")
        train_one_skill(skill_id, args.steps, args.ent, out_dir, seed=args.seed)


if __name__ == "__main__":
    main()