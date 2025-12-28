# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mini.env.v6_mini_env import MiniManagerEnv12, FlatMiniEnv12


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_low_policies(low_root: str, dims: int):
    from stable_baselines3 import PPO
    low_policies = []
    missing = []
    print(f"  开始加载底层策略，路径: {low_root}")
    
    for i in range(int(dims)):
        policy_path = os.path.join(low_root, f"skill{i}", "policy.zip")
        if os.path.exists(policy_path):
            try:
                policy = PPO.load(policy_path)
                low_policies.append(policy)
                print(f"  成功加载 skill{i}: {policy_path}")
            except Exception as e:
                print(f"  加载 skill{i} 失败: {e}")
                missing.append(policy_path)
        else:
            print(f"  文件不存在: {policy_path}")
            missing.append(policy_path)
    
    print(f"  总计加载 {len(low_policies)}/{dims} 个底层策略")
    
    if int(dims) == 12 and len(low_policies) != int(dims):
        print(f"  错误：缺失的策略文件: {missing}")
        raise FileNotFoundError(f"Expected 12 low-level PPO policies; missing: {missing}")
    
    return low_policies


def train_manager(steps: int, ent: float, out_dir: str, low_root: str, seed: int = 777, max_steps: int = 200, init_from: str | None = None, dims: int = 4, forgetting_mode: str = 'no_forgetting', resource_enabled: bool = False, resource_decay_range: tuple | list = (0.2, 0.4), match_bonus: float = 0.2, mismatch_penalty: float = 0.1, timing_bonus: float = 0.03, timing_penalty: float = 0.03, difficulty_bins: tuple | list = (0.33, 0.66), flat_mode: bool = False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    ensure_dir(out_dir)
    
    # 修复：预先加载低层策略，避免在环境创建时丢失
    low_policies = None
    if not flat_mode:
        low_policies = load_low_policies(low_root, dims)
        print(f"  成功加载 {len(low_policies)} 个低层策略用于训练")
    
    # 修复：创建环境工厂函数，确保low_policies正确传递
    def make_env():
        if flat_mode:
            return FlatMiniEnv12(
                tolerance=0.05,
                max_steps=max_steps,
                seed=seed,
                forgetting_mode=str(forgetting_mode),
                resource_enabled=bool(resource_enabled),
                resource_decay_range=tuple(resource_decay_range),
                match_bonus=float(match_bonus),
                mismatch_penalty=float(mismatch_penalty),
                timing_bonus=float(timing_bonus),
                timing_penalty=float(timing_penalty),
                difficulty_bins=tuple(difficulty_bins)
            )
        else:
            return MiniManagerEnv12(
                low_policies=low_policies,  # 使用预加载的策略
                tolerance=0.05,
                max_steps=max_steps,
                seed=seed,
                forgetting_mode=str(forgetting_mode),
                resource_enabled=bool(resource_enabled),
                resource_decay_range=tuple(resource_decay_range),
                match_bonus=float(match_bonus),
                mismatch_penalty=float(mismatch_penalty),
                timing_bonus=float(timing_bonus),
                timing_penalty=float(timing_penalty),
                difficulty_bins=tuple(difficulty_bins)
            )
    
    # 使用DummyVecEnv包装环境，确保兼容性
    env = DummyVecEnv([make_env])
    
    print(f"  环境创建成功，类型: DummyVecEnv")
    if not flat_mode:
        print(f"  环境中的低层策略数量: {len(low_policies)}")

    # 【修复22】添加训练回调监控坍缩
    class AntiCollapseCallback(BaseCallback):
        def __init__(self, check_freq=1000):
            super().__init__()
            self.check_freq = check_freq
            self.action_counts = np.zeros(env.envs[0].action_space.n)
            
        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                # 检查动作分布
                if len(self.model.ep_info_buffer) > 0:
                    recent_actions = self.model.rollout_buffer.actions[-100:] if hasattr(self.model.rollout_buffer, 'actions') else []
                    if len(recent_actions) > 0:
                        unique_actions = np.unique(recent_actions)
                        diversity = len(unique_actions) / env.envs[0].action_space.n
                        if diversity < 0.3:  # 如果动作多样性低于30%
                            print(f"警告：动作多样性低 ({diversity:.2%})，增加熵系数")
                            self.model.ent_coef = min(0.15, self.model.ent_coef * 1.5)
            return True

    class CurveCallback(BaseCallback):
        def __init__(self, log_path, forgetting_mode, seed):
            super().__init__()
            self.rows = []
            self.log_path = log_path
            self.forgetting_mode = forgetting_mode
            self.seed = seed
            # 确保目录存在
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        def _on_step(self) -> bool:
            if len(self.model.ep_info_buffer) > 0:
                # 基础指标
                ep_info = self.model.ep_info_buffer[-1]
                r = ep_info['r']
                l = ep_info['l']
                
                # 从info字典提取诊断指标
                info = ep_info.get('info', {})
                diag = {
                    'gap_variance': info.get('gap_variance', 0.0),
                    'action_entropy': info.get('action_entropy', 0.0),
                    'timer_mean': info.get('timer_mean', 0.0),
                    'timer_std': info.get('timer_std', 0.0),
                    'forgetting_triggers': info.get('forgetting_triggers', {}),
                    'effective_decay_cat': info.get('effective_decay_cat', {}),
                    'steps_since_review_cat': info.get('steps_since_review_cat', {}),
                    'action_freq': info.get('action_freq', []),
                    'forgetting_events': {cat: bool(v) for cat, v in info.get('forgetting_event', {}).items()}
                }
                
                # 合并记录
                self.rows.append({
                    "step": int(self.num_timesteps),
                    "ep_rew_mean": float(r),
                    "ep_len_mean": float(l),
                    "forgetting_mode": self.forgetting_mode,
                    "seed": self.seed,
                    **diag
                })
            return True
        
        def _on_training_end(self):
            # 保存完整训练曲线
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.rows, f, indent=2, ensure_ascii=False)
            
            # 自动运行最终评估
            self._run_final_evaluation()
        
        def _run_final_evaluation(self):
            """训练结束后自动运行50 episode评估"""
            env = self.training_env.envs[0]
            model = self.model
            
            eval_results = {
                'nf_eval': {'success_rate': 0, 'avg_steps': 0, 'metrics': {}},
                'ff_eval': {'success_rate': 0, 'avg_steps': 0, 'metrics': {}},
                'if_eval': {'success_rate': 0, 'avg_steps': 0, 'metrics': {}}
            }
            
            for eval_mode in ['no_forgetting', 'fixed_forgetting', 'improved_forgetting']:
                env.forgetting_mode = eval_mode
                successes = 0
                total_steps = 0
                metrics_accum = {
                    'gap_variance': [],
                    'forgetting_triggers': {'algebra':0,'geometry':0,'statistics':0}
                }
                
                for _ in range(50):
                    obs, info = env.reset()
                    done = False
                    steps = 0
                    
                    while not done and steps < env.max_steps:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        steps += 1
                    
                    successes += 1 if terminated else 0
                    total_steps += steps
                    
                    # 收集指标
                    metrics_accum['gap_variance'].append(info.get('gap_variance', 0.0))
                    for cat in ['algebra','geometry','statistics']:
                        metrics_accum['forgetting_triggers'][cat] += info.get('forgetting_triggers', {}).get(cat, 0)
                
                # 计算平均值
                key = f"{eval_mode[:2]}_eval"
                eval_results[key] = {
                    'success_rate': successes / 50.0,
                    'avg_steps': total_steps / 50.0,
                    'metrics': {
                        'avg_gap_variance': np.mean(metrics_accum['gap_variance']),
                        'total_forgetting_triggers': metrics_accum['forgetting_triggers']
                    }
                }
            
            eval_path = os.path.join(os.path.dirname(self.log_path), f"final_evaluation_{self.forgetting_mode}_seed{self.seed}.json")
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)

    # 【修复20】调整网络架构和训练参数
    if flat_mode:
        policy_kwargs = dict(net_arch=[256, 256])  # 减少一层
        effective_ent = max(0.30, ent)  # 【EXP15修复】极高熵，强制探索（扁平）
    else:
        policy_kwargs = dict(net_arch=[128, 128])  # 适中大小
        effective_ent = max(0.20, ent)  # 【EXP15修复】极高熵，强制探索（分层）

    if init_from and os.path.exists(init_from):
        # 修复：先创建模型，再加载参数，避免环境重新包装
        # 【修复21】使用更高的初始熵系数
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=2e-4,
            n_steps=512,          # 增加采样频率
            batch_size=128,
            n_epochs=8,
            gamma=0.99,
            ent_coef=effective_ent,  # 使用调整后的熵系数
            seed=seed,
            verbose=1,            # 显示训练日志
            policy_kwargs=policy_kwargs
        )
        
        # 加载预训练的参数
        pretrained_model = PPO.load(init_from)
        model.policy.load_state_dict(pretrained_model.policy.state_dict())
        print(f"  从 {init_from} 加载策略参数到新模型")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=2e-4,
            n_steps=512,          # 增加采样频率
            batch_size=128,
            n_epochs=8,
            gamma=0.99,
            ent_coef=effective_ent,  # 使用调整后的熵系数
            seed=seed,
            verbose=1,            # 显示训练日志
            policy_kwargs=policy_kwargs
        )
        print(f"  创建新的PPO模型")

    log_dir = os.path.join(out_dir, f"training_curve_{forgetting_mode}_seed{seed}.json")
    cb = CurveCallback(
        log_path=log_dir,
        forgetting_mode=forgetting_mode,
        seed=seed
    )
    
    # 【EXP15修复】单阶段训练，全程高熵，防止坍缩
    if flat_mode:
        # FLAT_IF模型：全程高熵
        stages = [
            (14500, effective_ent),  # 全程高熵
        ]
    else:
        # 分层模型：全程高熵
        stages = [
            (13000, effective_ent),  # 全程高熵
        ]
    
    total_trained = 0
    for stage_steps, stage_ent in stages:
        model.ent_coef = stage_ent
        print(f"训练阶段: {stage_steps}步, 熵系数={stage_ent:.3f}")
        model.learn(
            total_timesteps=stage_steps,
            callback=[AntiCollapseCallback(), cb],
            reset_num_timesteps=False
        )
        total_trained += stage_steps
        
        # 阶段评估
        if not flat_mode:
            # 简单评估
            env_single = env.envs[0]
            success_count = 0
            for _ in range(20):
                obs, _ = env_single.reset()
                done = False
                steps = 0
                while not done and steps < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env_single.step(action)
                    done = terminated or truncated
                    steps += 1
                success_count += 1 if terminated else 0
            
            success_rate = success_count / 20.0
            print(f"阶段完成，快速评估成功率: {success_rate:.2%}")
            if success_rate < 0.1 and total_trained > 5000:
                print("警告：成功率过低，考虑调整")

    # 创建结果目录结构
    mode_dir = os.path.join(out_dir, f"manager_{forgetting_mode}")
    ensure_dir(mode_dir)
    
    model.save(os.path.join(mode_dir, f"manager_policy_seed{seed}.zip"))
    
    print(f"训练完成！结果保存在: {mode_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--ent", type=float, default=0.02)
    parser.add_argument("--output", type=str, default="mini/results/manager")
    parser.add_argument("--low-root", type=str, default="mini/results/low")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-steps", type=int, default=800)  # 修改默认值为800
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--forgetting-mode", type=str, default='no_forgetting')
    parser.add_argument("--resource-enabled", action="store_true")
    parser.add_argument("--resource-decay-range", type=float, nargs=2, default=[0.2, 0.4])
    parser.add_argument("--match-bonus", type=float, default=0.2)
    parser.add_argument("--mismatch-penalty", type=float, default=0.1)
    parser.add_argument("--timing-bonus", type=float, default=0.03)
    parser.add_argument("--timing-penalty", type=float, default=0.03)
    parser.add_argument("--difficulty-bins", type=float, nargs=2, default=[0.33, 0.66])
    parser.add_argument("--flat-mode", action="store_true", help="Use Flat environment (no hierarchy)")
    args = parser.parse_args()

    ensure_dir(args.output)
    train_manager(
        args.steps,
        args.ent,
        args.output,
        args.low_root,
        seed=args.seed,
        max_steps=args.max_steps,
        init_from=args.init_from,
        dims=args.dims,
        forgetting_mode=args.forgetting_mode,
        resource_enabled=bool(args.resource_enabled),
        resource_decay_range=tuple(args.resource_decay_range),
        match_bonus=args.match_bonus,
        mismatch_penalty=args.mismatch_penalty,
        timing_bonus=args.timing_bonus,
        timing_penalty=args.timing_penalty,
        difficulty_bins=tuple(args.difficulty_bins),
        flat_mode=args.flat_mode
    )


if __name__ == "__main__":
    main()
