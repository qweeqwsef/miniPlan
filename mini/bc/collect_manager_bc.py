# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mini.env.v6_mini_env import MiniManagerEnv12, FlatMiniEnv12


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)


def load_low_policies(low_root: str, dims: int):
    from stable_baselines3 import PPO
    low_policies = []
    for i in range(dims):
        path = os.path.join(low_root, f"skill{i}", "policy.zip")
        if not os.path.exists(path):
            continue
        low_policies.append(PPO.load(path))
    return low_policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--output", type=str, default="mini/data/manager_bc.jsonl")
    parser.add_argument("--low-root", type=str, default="mini/results/low")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--forgetting-mode", type=str, default="no_forgetting")
    parser.add_argument("--resource-enabled", action="store_true")
    parser.add_argument("--flat-mode", action="store_true", help="Use Flat environment for BC collection")
    parser.add_argument("--max-steps", type=int, default=800, help="Maximum steps per episode")  # 【EXP15修复】确保专家策略充分轮换
    args = parser.parse_args()

    ensure_dir(args.output)
    
    if args.flat_mode:
        # Flat模式：不需要低层策略，直接使用FlatMiniEnv12
        env = FlatMiniEnv12(tolerance=0.05, max_steps=args.max_steps, seed=args.seed, 
                           forgetting_mode=str(args.forgetting_mode), 
                           resource_enabled=bool(args.resource_enabled))
    else:
        # 分层模式：需要低层策略
        low_policies = load_low_policies(args.low_root, args.dims)
        env = MiniManagerEnv12(low_policies=low_policies, tolerance=0.05, max_steps=args.max_steps, 
                              seed=args.seed, forgetting_mode=str(args.forgetting_mode), 
                              resource_enabled=bool(args.resource_enabled))

    count = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for _ in range(args.episodes):
            obs, info = env.reset()
            done = False
            while not done:
                # 【修复2】考虑Timer的专家策略
                if len(obs) > 12:
                    gaps = obs[:12]
                    timers = obs[-3:]  # 后3维是Timer
                else:
                    gaps = obs
                    timers = np.zeros(3)

                # 专家策略：90%贪婪+10%随机
                if np.random.random() < 0.9:
                    # Timer惩罚：Timer越高的类别，优先级越低
                    # 代数技能[0,1,2,3]使用timers[0]，几何[4,5,6,7]使用timers[1]，统计[8,9,10,11]使用timers[2]
                    timer_weights = np.array([
                        timers[0], timers[0], timers[0], timers[0],  # 代数
                        timers[1], timers[1], timers[1], timers[1],  # 几何
                        timers[2], timers[2], timers[2], timers[2]   # 统计
                    ])
                    
                    # 优先级 = gap - timer_weight * 0.3
                    priority = gaps - timer_weights * 0.3
                    skill_id = int(np.argmax(priority))
                    
                    if args.flat_mode:
                        action_type = int(np.random.randint(0, 6))
                        action = skill_id * 6 + action_type
                    else:
                        action = skill_id
                else:
                    action = env.action_space.sample()
                
                rec = {"obs": obs.tolist(), "action": int(action)}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                count += 1

    print(f"Saved {count} steps to {args.output}")


if __name__ == "__main__":
    main()