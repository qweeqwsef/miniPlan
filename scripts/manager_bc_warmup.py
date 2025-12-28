# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mini.env.v6_mini_env import MiniManagerEnv12, FlatMiniEnv12


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_low_policies(low_root: str, dims: int):
    from stable_baselines3 import PPO
    low_policies = []
    for i in range(dims):
        path = os.path.join(low_root, f"skill{i}", "policy.zip")
        if not os.path.exists(path):
            continue
        low_policies.append(PPO.load(path))
    return low_policies


def load_dataset(jsonl_path: str):
    X, y = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            X.append(np.array(rec["obs"], dtype=np.float32))
            y.append(int(rec["action"]))
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y


def supervised_warmup(model, X: np.ndarray, y: np.ndarray, epochs: int = 8, lr: float = 3e-4, batch_size: int = 256, log_path: str = None):
    """
    改进3: BC decay - BC loss系数随训练步数衰减至0
    增强数据记录：记录BC训练过程中的策略演化数据
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.policy.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = model.policy.optimizer if hasattr(model.policy, "optimizer") else optim.Adam(model.policy.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.policy.train()
    
    # 改进3: 计算总训练步数用于BC decay
    total_steps = len(loader) * epochs
    bc_decay_steps = total_steps  # BC在所有步数内衰减至0
    initial_bc_coef = 1.0
    
    # 增强数据记录：BC训练过程记录
    bc_evolution = []
    
    step_count = 0
    for epoch in range(epochs):
        # 动态获取动作空间大小
        action_space_size = model.env.action_space.n if hasattr(model.env, 'action_space') else 12
        
        epoch_metrics = {
            'epoch': epoch,
            'loss': 0.0,
            'accuracy': 0.0,
            'action_distribution': np.zeros(action_space_size)
        }
        
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device)
            
            # 改进3: 计算BC系数衰减
            bc_coef = initial_bc_coef * max(0.0, 1.0 - step_count / bc_decay_steps)
            
            # 前向：提取特征 → pi → logits
            features = model.policy.extract_features(xb)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)
            loss = criterion(logits, yb) * bc_coef  # 应用衰减系数
            
            # 计算准确率和动作分布
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                accuracy = (pred == yb).float().mean().item()
                
                # 记录动作分布
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                epoch_metrics['action_distribution'] += np.sum(probs, axis=0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['accuracy'] += accuracy
            
            # 记录详细的训练数据
            if step_count % 10 == 0:  # 每10步记录一次
                bc_evolution.append({
                    "step": step_count,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "bc_coef": bc_coef,
                    "loss": loss.item(),
                    "accuracy": accuracy,
                    "lr": optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else lr
                })
            
            step_count += 1
        
        # 归一化分布和计算epoch平均值
        epoch_metrics['action_distribution'] = (
            epoch_metrics['action_distribution'] / 
            np.sum(epoch_metrics['action_distribution'])
        ).tolist()
        epoch_metrics['loss'] /= len(loader)
        epoch_metrics['accuracy'] /= len(loader)
        epoch_metrics['final_bc_coef'] = bc_coef
        epoch_metrics['total_batches'] = len(loader)
        
        bc_evolution.append(epoch_metrics)
        
        print(f"BC Epoch {epoch+1}/{epochs}: Loss={epoch_metrics['loss']:.4f}, Acc={epoch_metrics['accuracy']:.3f}, BC_coef={bc_coef:.3f}")
    
    # 保存BC演化数据
    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(bc_evolution, f, indent=2, ensure_ascii=False)
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-data", type=str, default="mini/data/manager_bc.jsonl")
    parser.add_argument("--low-root", type=str, default="mini/results/low")
    parser.add_argument("--output", type=str, default="mini/results/manager_bc_init")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--resource-enabled", action="store_true")
    parser.add_argument("--forgetting-mode", type=str, default='no_forgetting')
    parser.add_argument("--flat-mode", action="store_true", help="Use Flat environment for BC warmup")
    args = parser.parse_args()

    ensure_dir(args.output)
    X, y = load_dataset(args.bc_data)

    dims_skills = int(args.dims)
    
    if args.flat_mode:
        # Flat模式：不需要低层策略，直接使用FlatMiniEnv12
        env = FlatMiniEnv12(tolerance=0.05, max_steps=800, seed=args.seed, 
                           forgetting_mode=str(args.forgetting_mode), 
                           resource_enabled=bool(args.resource_enabled))
    else:
        # 分层模式：需要低层策略
        low_policies = load_low_policies(args.low_root, dims_skills)
        env = MiniManagerEnv12(low_policies=low_policies, tolerance=0.05, max_steps=800, 
                              seed=args.seed, forgetting_mode=str(args.forgetting_mode), 
                              resource_enabled=bool(args.resource_enabled))

    from stable_baselines3 import PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        seed=args.seed,
        verbose=0,
        policy_kwargs=dict(net_arch=[128, 128])  # 与管理者训练保持一致
    )

    # 创建结果目录结构
    mode_dir = os.path.join(args.output, f"manager_{args.forgetting_mode}")  # 统一目录结构
    ensure_dir(mode_dir)
    
    bc_log_path = os.path.join(args.output, f"manager_{args.forgetting_mode}", f"bc_evolution_seed{args.seed}.json")  # 与训练曲线同目录
    supervised_warmup(model, X, y, epochs=8, lr=3e-4, batch_size=256, log_path=bc_log_path)
    
    out_path = os.path.join(mode_dir, f"manager_bc_seed{args.seed}.zip")
    model.save(out_path)
    
    # 生成BC阶段汇总
    bc_summary = {
        "forgetting_mode": args.forgetting_mode,
        "seed": args.seed,
        "total_epochs": 8,
        "total_samples": len(X),
        "bc_decay_applied": True,
        "flat_mode": args.flat_mode
    }
    
    summary_file = os.path.join(mode_dir, f"bc_summary_{args.forgetting_mode}_seed{args.seed}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(bc_summary, f, indent=2, ensure_ascii=False)
    
    print(f"BC预热完成！模型保存至: {out_path}")
    print(f"训练日志保存至: {bc_log_path}")


if __name__ == "__main__":
    main()
