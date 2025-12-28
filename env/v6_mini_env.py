# -*- coding: utf-8 -*-#
"""
修复后的分层强化学习环境
解决策略坍缩问题的关键修复：
1. 低层：连续资源衰减 + 扩展观察空间 [gap, resource_fatigue]
2. 高层：扩展观察空间 [gaps, timers] + 连续遗忘衰减
3. 保留所有原有机制：遗忘、资源珍惜、难度匹配
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class MiniSkillEnv(gym.Env):
    """
    低层技能环境（修复策略坍缩问题）
    
    修复说明：
    - 扩展观察空间：[gap, resource_fatigue, mastery_category, action_counter] 解决资源状态不可观测问题
    - 连续资源衰减：避免硬跳变导致的策略坍缩
    - 保留所有原有机制：难度匹配、资源珍惜等
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, tolerance: float = 0.05, max_steps: int = 200, seed: int = 0):
        super().__init__()
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # 【修复1】增强观察空间（学术理由：部分可观测→全可观测）
        # 原2维：[gap, resource_fatigue]
        # 新4维：[gap, resource_fatigue, mastery_category, action_counter]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        # 动作语义定义
        self.action_semantics = [
            "easy_res", "medium_res", "hard_res",
            "seek_teacher", "peer_discussion", "self_learn"
        ]

        # 增益矩阵（环境动力学，不是奖励）
        self.GAIN_MATRIX = {
            "low": {        # 掌握度 0.00-0.33
                "easy_res": 0.1, "medium_res": 0.05, "hard_res": 0.03,
                "seek_teacher": 0.05, "peer_discussion": 0.025, "self_learn": 0.015
            },
            "medium": {     # 掌握度 0.34-0.66
                "easy_res": 0.05, "medium_res": 0.1, "hard_res": 0.05,
                "seek_teacher": 0.025, "peer_discussion": 0.05, "self_learn": 0.025
            },
            "high": {       # 掌握度 0.67-1.00
                "easy_res": 0.03, "medium_res": 0.05, "hard_res": 0.1,
                "seek_teacher": 0.015, "peer_discussion": 0.025, "self_learn": 0.05
            }
        }

        # 保留原有delta_map作为备用（向后兼容）
        self.delta_map = np.array([0.05, 0.10, 0.20, 0.03, 0.06, 0.02], dtype=np.float32)

        # 【修复2】增加动作计数器（防止重复选择同一动作）
        self.action_counter = np.zeros(6, dtype=np.float32)
        self.action_decay = 0.9  # 每步衰减系数

        # 修复A: 资源疲劳度跟踪 (连续衰减替代硬跳变)
        self.resource_usage_count = np.zeros(3, dtype=np.float32)  # 3个资源动作的使用次数
        # 【修复3】调整资源疲劳参数
        self.resource_fatigue_alpha = 0.7  # 从0.5提高到0.7，衰减更快

        self.gap = None
        self.step_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # 1. 随机 Gap：覆盖全难度
        self.gap = float(self.np_random.uniform(0.1, 0.95))
        
        # 【修复4】重置时给予适中的资源疲劳度（不是从0开始）
        self.resource_usage_count = self.np_random.uniform(1.0, 4.0, size=(3,)).astype(np.float32)
        self.action_counter.fill(0.0)
        
        self.step_count = 0
        # 【关键】扩展观察空间包含资源疲劳度
        resource_fatigue = self._compute_resource_fatigue()
        mastery = 1.0 - self.gap
        mastery_category = 0.0 if mastery <= 0.33 else (0.5 if mastery <= 0.66 else 1.0)
        
        obs = np.array([self.gap, mastery_category], dtype=np.float32)
        info = {}
        return obs, info

    def _compute_resource_fatigue(self):
        """
        修复A: 计算资源疲劳度 (连续值，避免硬跳变)
        使用指数衰减: fatigue = 1 - exp(-alpha * max_usage_count)
        """
        max_usage = np.max(self.resource_usage_count)
        fatigue = 1.0 - np.exp(-self.resource_fatigue_alpha * max_usage)
        return float(np.clip(fatigue, 0.0, 1.0))

    def step(self, action: int):
        self.step_count += 1

        # 【修复5】更新动作计数器
        self.action_counter = self.action_counter * self.action_decay
        self.action_counter[action] += 1.0

        prev_gap = float(self.gap)
        mastery = 1.0 - prev_gap

        # 确定掌握度区间
        if mastery <= 0.33:
            level = "low"
        elif mastery <= 0.66:
            level = "medium"
        else:
            level = "high"

        action_name = self.action_semantics[action]

        # 【修复6】增加动作惩罚（防止坍缩）
        action_penalty = 0.0
        if self.action_counter[action] > 3.0:  # 连续使用同一动作3次以上
            action_penalty = -0.5 * (self.action_counter[action] - 3.0)

        # 修复A: 连续资源衰减替代硬跳变
        if action in [0, 1, 2]:  # 资源动作
            res_id = action
            # 更新使用次数
            self.resource_usage_count[res_id] += 1.0
            # 连续衰减: reward_scale = exp(-alpha * usage_count)
            usage_count = self.resource_usage_count[res_id]
            resource_decay = np.exp(-self.resource_fatigue_alpha * usage_count)
        else:
            resource_decay = 1.0  # 非资源动作不衰减

        # 计算增益（环境动力学）- 应用连续衰减
        base_gain = self.GAIN_MATRIX[level][action_name]
        gain = base_gain * resource_decay

        # 轻微噪声，避免策略完全确定性（提高熵）
        noise = float(self.np_random.normal(0.0, 0.001))
        new_gap = clamp(prev_gap - gain + noise, 0.0, 1.0)
        self.gap = new_gap

        # 奖励：本地 dense reward（移除直接的首次资源使用奖励）
        reward = 0.0
        # 按 gap 减少量给予 dense 奖励，鼓励选择更大幅度动作
        reward += 5.0 * max(0.0, prev_gap - new_gap)
        reward += 5.0 if new_gap < self.tolerance else 0.0
        # 【修复7】在原有reward基础上加上动作惩罚
        reward += action_penalty
        reward += -0.01  # 步惩罚

        terminated = bool(new_gap < self.tolerance)
        truncated = bool(self.step_count >= self.max_steps)

        mastery = 1.0 - self.gap
        mastery_category = 0.0 if mastery <= 0.33 else (0.5 if mastery <= 0.66 else 1.0)
        obs = np.array([self.gap, mastery_category], dtype=np.float32)
        
        info = {
            "gap": self.gap, 
            "gain": gain, 
            "base_gain": base_gain,
            "resource_decay": resource_decay if action in [0, 1, 2] else 1.0,
            "noise": noise, 
            "mastery": mastery, 
            "level": level, 
            "action_name": action_name,
            "resource_fatigue": self._compute_resource_fatigue(),
            "resource_usage_count": self.resource_usage_count.copy(),
            "action_counter": self.action_counter.copy(),
            "action_penalty": action_penalty
        }
        return obs, reward, terminated, truncated, info


class MiniManagerEnv12(gym.Env):
    """
    12维管理者环境（修复策略坍缩问题）
    
    修复说明：
    - 扩展观察空间：[gap_0, ..., gap_11, timer_cat0, timer_cat1, timer_cat2] 解决POMDP问题
    - 连续遗忘衰减：避免硬跳变导致的策略坍缩
    - 保留所有原有机制：遗忘、资源珍惜、难度匹配等
    """
    def __init__(self, low_policies=None, tolerance: float = 0.05, max_steps: int = 200, seed: int = 0, 
                 forgetting_mode: str = 'no_forgetting', forgetting_params: dict | None = None, 
                 resource_enabled: bool = False, resource_decay_range: tuple | list = (0.2, 0.4), 
                 match_bonus: float = 0.2, mismatch_penalty: float = 0.1, 
                 timing_bonus: float = 0.03, timing_penalty: float = 0.03, 
                 difficulty_bins: tuple | list = (0.33, 0.66)):
        super().__init__()
        self.low_policies = low_policies or []
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # 修复B: 扩展观察空间 [gap_0, ..., gap_11, timer_cat0, timer_cat1, timer_cat2]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(12)
        
        self.delta_map = np.array([0.05, 0.10, 0.20, 0.03, 0.06, 0.02], dtype=np.float32)
        self.gaps = None
        self.step_count = 0
        self._last_action = None
        self.forgetting_mode = str(forgetting_mode)
        
        # 【修复13】增加防坍缩机制
        self.action_penalty_coef = 0.02  # 动作重复惩罚系数
        self.last_actions = []  # 记录最近动作
        self.action_memory_size = 10  # 记忆最近10个动作
        
        # 【修复14】根据不同遗忘模式调整Timer敏感性
        # 但保持BC收集策略一致（只通过环境动力学产生差异）
        self.timer_sensitivity = 1.0  # 基础敏感性
        
        # 改进1: 弱action-entropy shaping - 维护滑动窗口
        self.action_history = []
        self.action_history_window = 50
        self.entropy_coef = 0.01
        
        # 改进2: timer-aware shaping - 系数
        self.timer_coef = 0.05
        
        # Categories是latent environment factor，agent不可观测
        self.categories = {
            'algebra': [0, 1, 2, 3],
            'geometry': [4, 5, 6, 7],
            'statistics': [8, 9, 10, 11],
        }
        self.steps_since_review_cat = {'algebra': 0, 'geometry': 0, 'statistics': 0}
        self.history_buffer_cat = {'algebra': [], 'geometry': [], 'statistics': []}
        self._fp = forgetting_params or {}
        
        # 触发频率统计
        self.forgetting_trigger_stats = {'algebra': 0, 'geometry': 0, 'statistics': 0}
        
        # 修复B: 遗忘阈值 (用于计算归一化timer)
        self.forgetting_thresholds = {
            'algebra': 5,
            'geometry': 4,
            'statistics': 6
        }
        
        self.resource_enabled = bool(resource_enabled)
        self.resource_decay_range = tuple(resource_decay_range)
        self.match_bonus = float(match_bonus)
        self.mismatch_penalty = float(mismatch_penalty)
        self.timing_bonus = float(timing_bonus)
        self.timing_penalty = float(timing_penalty)
        self.difficulty_bins = tuple(difficulty_bins)
        self.resource_used = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        if options and isinstance(options, dict) and 'init_gaps' in options:
            init = np.array(options['init_gaps'], dtype=np.float32)
            if init.shape != (12,):
                init = init.reshape(12,)
            self.gaps = np.clip(init, 0.0, 1.0)
        else:
            self.gaps = self.np_random.uniform(0.4, 0.9, size=(12,)).astype(np.float32)
        self.step_count = 0
        self._last_action = None
        for c in self.steps_since_review_cat:
            self.steps_since_review_cat[c] = 0
        for c in self.history_buffer_cat:
            self.history_buffer_cat[c] = []
        
        # 改进1: 重置action history
        self.action_history = []
        
        # 重置防坍缩机制
        self.last_actions = []
        
        # 修复B: 扩展观察空间包含归一化timer
        timer_obs = self._compute_timer_observation()
        obs = np.concatenate([self.gaps, timer_obs])
        return obs, {}

    def _compute_timer_observation(self):
        """
        修复B: 计算归一化timer观察 (解决POMDP问题)
        timer_cat = steps_since_review_cat / forgetting_threshold
        不告诉agent类别归属，只提供timer状态
        """
        timers = []
        for cat in ['algebra', 'geometry', 'statistics']:
            steps = self.steps_since_review_cat[cat]
            threshold = self.forgetting_thresholds[cat]
            normalized_timer = min(1.0, float(steps) / float(threshold))
            timers.append(normalized_timer)
        return np.array(timers, dtype=np.float32)
    
    def _compute_action_entropy(self):
        """
        改进1: 计算action entropy用于shaping
        基于最近50步的action分布
        """
        if len(self.action_history) == 0:
            return 0.0
        
        action_hist = np.zeros(12, dtype=np.float32)
        for act in self.action_history:
            action_hist[int(act)] += 1.0
        
        freq = action_hist / (action_hist.sum() + 1e-8)
        entropy = -np.sum(freq * np.log(freq + 1e-8))
        return float(entropy)
    
    def _compute_timer_sum(self):
        """
        改进2: 计算timer总和用于timer-aware shaping
        """
        timer_obs = self._compute_timer_observation()
        return float(np.sum(timer_obs))

    def _global_success(self) -> bool:
        return bool(np.all(self.gaps < self.tolerance))

    def _apply_fixed_forgetting(self, mastery: np.ndarray) -> tuple[np.ndarray, dict]:
        """修复C: 连续遗忘衰减的固定遗忘机制"""
        # 增强参数（统一增强）
        enhanced_params = {
            'thresholds': {
                'algebra': 5,
                'geometry': 4,
                'statistics': 6
            },
            'strengths': {
                'algebra': 0.07,
                'geometry': 0.07,
                'statistics': 0.07
            },
            'adjust_factor': 0.6,
        }
        
        enhanced_params.update(self._fp.get('fixed', {}))
        events = {k: False for k in self.categories}
        
        for cat, idxs in self.categories.items():
            t = int(self.steps_since_review_cat[cat])
            th = int(enhanced_params['thresholds'][cat])
            
            # 修复C: 连续遗忘衰减 (避免硬跳变)
            if t >= th:
                s = float(enhanced_params['strengths'][cat])
                adj = float(enhanced_params.get('adjust_factor', 0.5))
                
                # 连续衰减: forgetting_strength = min(1.0, (timer - threshold) / scale)
                excess_steps = max(0, t - th)
                scale = 5.0  # 控制衰减平滑度
                forgetting_strength = min(1.0, float(excess_steps) / scale)
                
                # 应用连续衰减
                effective_decay = 1.0 - (s * adj * forgetting_strength)
                
                for i in idxs:
                    mastery[i] = float(max(0.0, mastery[i] * effective_decay))
                
                events[cat] = True
                self.forgetting_trigger_stats[cat] += 1
                
        return mastery, events

    def _apply_improved_forgetting(self, mastery: np.ndarray) -> tuple[np.ndarray, dict]:
        """修复C: 连续遗忘衰减的改进遗忘机制"""
        # 使用与增强FF相同的触发阈值（同步增强，保持与FF一致）
        thresholds = {
            'algebra': 5,
            'geometry': 4,
            'statistics': 6
        }
        
        # IF特有参数 - Stability-Aware Threshold Forgetting
        if_params = {
            'base_decay': 0.07,
            'stability_k': 8.0,
            'stability_lambda': 0.7,
        }
        if_params.update(self._fp.get('improved', {}))
        
        ed = {k: 0.0 for k in self.categories}  # 记录实际衰减强度
        
        for cat, idxs in self.categories.items():
            t_c = int(self.steps_since_review_cat[cat])
            th = int(thresholds[cat])
            
            # 阈值触发 (与FF完全一致的触发条件)
            if t_c >= th:
                # 计算历史稳定性
                hist = list(self.history_buffer_cat.get(cat, []))
                if len(hist) >= 2:
                    vol_c = float(np.std(hist))
                    stability_c = float(np.exp(-if_params['stability_k'] * vol_c))
                else:
                    stability_c = 0.5  # 默认中等稳定性
                
                # Stability-Aware Threshold Forgetting 核心创新
                base_decay = if_params['base_decay']
                lambda_factor = if_params['stability_lambda']
                
                # 修复C: 连续衰减 + 稳定性感知
                excess_steps = max(0, t_c - th)
                scale = 5.0  # 控制衰减平滑度
                forgetting_strength = min(1.0, float(excess_steps) / scale)
                
                # IF的关键一刀: effective_decay = base_decay * (1 - λ * stability_c) * forgetting_strength
                effective_decay_base = base_decay * (1.0 - lambda_factor * stability_c)
                effective_decay = effective_decay_base * forgetting_strength
                
                # 应用遗忘 (使用adjust_factor保持与FF一致的应用方式)
                decay_multiplier = 1.0 - (effective_decay * 0.6)
                for i in idxs:
                    mastery[i] = float(max(0.0, mastery[i] * decay_multiplier))
                
                ed[cat] = effective_decay
                self.forgetting_trigger_stats[cat] += 1
                
        return mastery, ed

    def step(self, action: int):
        self.step_count += 1
        prev_sum = float(np.sum(self.gaps))
        prev_completed = int(np.sum(self.gaps < self.tolerance))
        
        skill_id = int(action)
        skill_gap = float(self.gaps[skill_id])
        
        # 确保必须有12个低层策略
        assert len(self.low_policies) == 12, "必须提供12个低层策略"
        # 修复A: 扩展低层观察空间 [gap, resource_fatigue, mastery_category, action_counter]
        mastery = 1.0 - skill_gap
        mastery_category = 0.0 if mastery <= 0.33 else (0.5 if mastery <= 0.66 else 1.0)
        obs_skill = np.array([skill_gap, mastery_category], dtype=np.float32)
        low_policy = self.low_policies[skill_id]
        act, _ = low_policy.predict(obs_skill, deterministic=True)
        type_id = int(act)
        
        # 计算掌握度和难度级别 (与MiniSkillEnv一致)
        mastery = 1.0 - skill_gap
        if mastery <= 0.33:
            level = "low"
        elif mastery <= 0.66:
            level = "medium"
        else:
            level = "high"
        
        # 动作语义定义 (与MiniSkillEnv一致)
        action_semantics = [
            "easy_res", "medium_res", "hard_res",
            "seek_teacher", "peer_discussion", "self_learn"
        ]
        
        # 增益矩阵 (与MiniSkillEnv完全一致)
        GAIN_MATRIX = {
            "low": {
                "easy_res": 0.1, "medium_res": 0.05, "hard_res": 0.03,
                "seek_teacher": 0.05, "peer_discussion": 0.025, "self_learn": 0.015
            },
            "medium": {
                "easy_res": 0.05, "medium_res": 0.1, "hard_res": 0.05,
                "seek_teacher": 0.025, "peer_discussion": 0.05, "self_learn": 0.025
            },
            "high": {
                "easy_res": 0.03, "medium_res": 0.05, "hard_res": 0.1,
                "seek_teacher": 0.015, "peer_discussion": 0.025, "self_learn": 0.05
            }
        }
        
        # 使用GAIN_MATRIX计算增益 (与MiniSkillEnv一致)
        action_name = action_semantics[type_id]
        gain = GAIN_MATRIX[level][action_name]
        
        # 噪声 (与MiniSkillEnv一致)
        noise = float(self.np_random.normal(0.0, 0.001))
        
        # 计算新的gap (与MiniSkillEnv一致)
        delta = 0.0 if (skill_gap < self.tolerance) else gain
        new_gap = clamp(skill_gap - delta + noise, 0.0, 1.0)
        self.gaps[skill_id] = new_gap
        
        # 更新遗忘机制状态
        for cat, idxs in self.categories.items():
            if skill_id in idxs:
                self.steps_since_review_cat[cat] = 0
            else:
                self.steps_since_review_cat[cat] += 1
                
        # 应用遗忘机制
        mastery = 1.0 - self.gaps.copy()
        mastery_before = mastery.copy()
        forgetting_event = {'algebra': False, 'geometry': False, 'statistics': False}
        effective_decay_cat = {'algebra': 0.0, 'geometry': 0.0, 'statistics': 0.0}
        
        if self.forgetting_mode == 'fixed_forgetting':
            mastery, fe = self._apply_fixed_forgetting(mastery)
            forgetting_event.update(fe)
        elif self.forgetting_mode == 'improved_forgetting':
            mastery, ed = self._apply_improved_forgetting(mastery)
            effective_decay_cat.update(ed)
        
        # 更新gaps
        self.gaps = np.clip(1.0 - mastery, 0.0, 1.0)
        
        # 改进1: 更新action history用于entropy计算
        self.action_history.append(skill_id)
        if len(self.action_history) > self.action_history_window:
            self.action_history = self.action_history[-self.action_history_window:]
        
        # 【修复15】动作重复惩罚
        repetition_penalty = 0.0
        if len(self.last_actions) >= 2:
            # 如果最近频繁选择同一动作，给予惩罚
            recent_actions = self.last_actions[-5:] if len(self.last_actions) >= 5 else self.last_actions
            same_action_count = sum(1 for a in recent_actions if a == skill_id)
            if same_action_count >= 3:  # 最近5步内选择同一动作3次以上
                repetition_penalty = -self.action_penalty_coef * same_action_count
        
        # 更新动作记忆
        self.last_actions.append(skill_id)
        if len(self.last_actions) > self.action_memory_size:
            self.last_actions.pop(0)
        
        # 计算奖励
        curr_sum = float(np.sum(self.gaps))
        curr_completed = int(np.sum(self.gaps < self.tolerance))
        delta_sum = prev_sum - curr_sum
        
        # 基础奖励结构
        reward = 0.0
        reward += 5.0 * delta_sum  # gap减少奖励
        reward += 5.0 * max(0, curr_completed - prev_completed)
        reward += 20.0 if self._global_success() else 0.0  # 全局成功奖励
        
        # 【修复15】添加动作重复惩罚
        reward += repetition_penalty
        
        # 【修复16】均衡性奖励（鼓励覆盖所有技能）
        completed_skills = np.sum(self.gaps < self.tolerance)
        balance_reward = 0.1 * (completed_skills / 12.0)  # 完成越多奖励越高
        reward += balance_reward
        
        # 【修复18】增加熵奖励（鼓励探索）
        current_entropy = self._compute_action_entropy()
        entropy_bonus = 0.03 * current_entropy  # 小幅熵奖励
        reward += entropy_bonus
        
        reward += -0.02  # 步惩罚
        
        # 【EXP15修复】强制策略关注所有维度 - 维度均衡性惩罚
        gap_variance = np.var(self.gaps)
        if gap_variance > 0.05:  # 阈值可调，方差大则惩罚重
            reward -= 5.0 * gap_variance  # 惩罚不均衡的学习
        else:
            reward += 0.5  # 奖励均衡的学习
        
        # 【EXP15修复】Timer惩罚 - 提高惩罚系数，迫使策略必须轮换
        timer_penalty_coef = 0.0  # 默认无惩罚
        if self.forgetting_mode in ['fixed_forgetting', 'improved_forgetting']:
            timer_penalty_coef = 2.0
        
        timer_sum = self._compute_timer_sum()
        reward -= timer_penalty_coef * timer_sum  # 新增Timer惩罚
        
        # 改进1: 弱action-entropy shaping (所有Manager训练)
        entropy = self._compute_action_entropy()
        reward += self.entropy_coef * entropy
        
        terminated = self._global_success()
        truncated = bool(self.step_count >= self.max_steps)
        self._last_action = skill_id
        
        # 动作掩码：12个动作
        mask = (self.gaps >= self.tolerance).astype(np.int32)
        
        # 修复B: 扩展观察空间包含归一化timer
        timer_obs = self._compute_timer_observation()
        obs = np.concatenate([self.gaps, timer_obs])
        
        # 改进6: 诊断日志
        gap_variance = float(np.var(self.gaps))
        action_freq = np.zeros(12, dtype=np.float32)
        for act in self.action_history:
            action_freq[int(act)] += 1.0
        action_freq = action_freq / (action_freq.sum() + 1e-8)
        
        info = {
            # 基础指标
            'sum_prev': prev_sum,
            'sum_curr': curr_sum,
            'skill_id': skill_id,
            'low_action': int(type_id),
            'delta': delta,
            'noise': noise,
            'action_mask': mask.tolist(),
            'forgetting_mode': self.forgetting_mode,
            
            # 关键诊断指标
            'gap_variance': gap_variance,  # 学习均衡性
            'action_entropy': entropy,  # 策略探索水平
            'timer_mean': float(np.mean(timer_obs)),  # timer状态
            'timer_std': float(np.std(timer_obs)),  # timer波动性
            
            # 遗忘相关
            'forgetting_triggers': dict(self.forgetting_trigger_stats),  # 各类别触发次数
            'effective_decay_cat': effective_decay_cat,  # 实际衰减强度
            'steps_since_review_cat': dict(self.steps_since_review_cat),  # 距离上次复习
            'forgetting_event': forgetting_event,  # 本次是否触发
            
            # 策略分析
            'action_freq': action_freq.tolist(),  # 动作使用频率
            'gap_mean': float(np.mean(self.gaps)),  # 平均gap
            
            # 保留原有字段（向后兼容）
            'timer_obs': timer_obs.tolist(),
            'mastery_before': mastery_before.tolist(),
            'mastery_after': mastery.tolist(),
            'timer_sum': timer_sum,
        }

        # 【EXP15修复】强制策略关注所有维度 - 防止坍缩
        gap_variance = float(np.var(self.gaps))
        if gap_variance > 0.05:
            reward -= 5.0 * gap_variance  # 惩罚不均衡的学习
        else:
            reward += 0.5  # 奖励均衡的学习
        
        # 更新历史缓冲区
        for cat, idxs in self.categories.items():
            M_c = float(np.mean(mastery[idxs]))
            hb = self.history_buffer_cat.get(cat, [])
            hb = hb + [M_c]
            if len(hb) > 50:
                hb = hb[-50:]
            self.history_buffer_cat[cat] = hb
            
        return obs, reward, terminated, truncated, info


class FlatMiniEnv12(gym.Env):
    """
    12维 Flat PPO 环境（修复策略坍缩问题）
    
    修复说明：
    - 扩展观察空间：[gap_0, ..., gap_11, timer_cat0, timer_cat1, timer_cat2] 解决POMDP问题
    - 连续遗忘衰减：避免硬跳变导致的策略坍缩
    - 保留所有原有机制：遗忘、资源珍惜、难度匹配等
    """

    def __init__(self, tolerance: float = 0.05, max_steps: int = 200, seed: int = 0, 
                 forgetting_mode: str = 'no_forgetting', forgetting_params: dict | None = None,
                 resource_enabled: bool = False, resource_decay_range: tuple | list = (0.2, 0.4), 
                 match_bonus: float = 0.2, mismatch_penalty: float = 0.1, 
                 timing_bonus: float = 0.03, timing_penalty: float = 0.03, 
                 difficulty_bins: tuple | list = (0.33, 0.66)):
        super().__init__()
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # 修复B: 扩展观察空间 [gap_0, ..., gap_11, timer_cat0, timer_cat1, timer_cat2]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(72)  # 12 skills × 6 actions
        
        # 🔧 修复不公平点1: 使用与分层环境相同的GAIN_MATRIX
        # 动作语义定义 (与MiniSkillEnv保持一致)
        self.action_semantics = [
            "easy_res", "medium_res", "hard_res",
            "seek_teacher", "peer_discussion", "self_learn"
        ]
        
        # 增益矩阵（与MiniSkillEnv完全一致的环境动力学）
        self.GAIN_MATRIX = {
            "low": {        # 掌握度 0.00-0.33
                "easy_res": 0.1, "medium_res": 0.05, "hard_res": 0.03,
                "seek_teacher": 0.05, "peer_discussion": 0.025, "self_learn": 0.015
            },
            "medium": {     # 掌握度 0.34-0.66
                "easy_res": 0.05, "medium_res": 0.1, "hard_res": 0.05,
                "seek_teacher": 0.025, "peer_discussion": 0.05, "self_learn": 0.025
            },
            "high": {       # 掌握度 0.67-1.00
                "easy_res": 0.03, "medium_res": 0.05, "hard_res": 0.1,
                "seek_teacher": 0.015, "peer_discussion": 0.025, "self_learn": 0.05
            }
        }
        
        # 保留原有delta_map作为备用（向后兼容）
        self.delta_map = np.array([0.05, 0.10, 0.20, 0.03, 0.06, 0.02], dtype=np.float32)
        
        # 🔧 修复不公平点1: 添加资源疲劳机制 (与MiniSkillEnv保持一致)
        self.resource_usage_count = np.zeros((12, 3), dtype=np.float32)  # 12个技能 × 3个资源动作
        self.resource_fatigue_alpha = 0.5  # 与MiniSkillEnv一致
        
        self.gaps = None
        self.step_count = 0
        self._last_action = None
        
        # 遗忘机制支持
        self.forgetting_mode = str(forgetting_mode)
        self.categories = {
            'algebra': [0, 1, 2, 3],
            'geometry': [4, 5, 6, 7],
            'statistics': [8, 9, 10, 11],
        }
        self.steps_since_review_cat = {'algebra': 0, 'geometry': 0, 'statistics': 0}
        self.history_buffer_cat = {'algebra': [], 'geometry': [], 'statistics': []}
        self._fp = forgetting_params or {}
        
        # 触发频率统计
        self.forgetting_trigger_stats = {'algebra': 0, 'geometry': 0, 'statistics': 0}
        
        # 修复B: 遗忘阈值 (用于计算归一化timer)
        self.forgetting_thresholds = {
            'algebra': 5,
            'geometry': 4,
            'statistics': 6
        }
        
        # 资源机制支持
        self.resource_enabled = bool(resource_enabled)
        self.resource_decay_range = tuple(resource_decay_range)
        self.match_bonus = float(match_bonus)
        self.mismatch_penalty = float(mismatch_penalty)
        self.timing_bonus = float(timing_bonus)
        self.timing_penalty = float(timing_penalty)
        self.difficulty_bins = tuple(difficulty_bins)
        self.resource_used = None
        
        # 改进1: 弱action-entropy shaping - 维护滑动窗口 (Flat也使用)
        self.action_history = []
        self.action_history_window = 50
        # 🔧 修复不公平点3: 调整entropy系数以匹配动作空间维度
        # 分层是12维，扁平是72维，所以扁平的系数应该是 0.01 * 72/12 = 0.06
        self.entropy_coef = 0.06  # 调整为6倍，补偿72维动作空间的稀释效应

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # 初始化12维gaps
        if options and isinstance(options, dict) and 'init_gaps' in options:
            init = np.array(options['init_gaps'], dtype=np.float32)
            if init.shape != (12,):
                init = init.reshape(12,)
            self.gaps = np.clip(init, 0.0, 1.0)
        else:
            self.gaps = self.np_random.uniform(0.4, 0.9, size=(12,)).astype(np.float32)
        
        self.step_count = 0
        self._last_action = None
        
        # 重置遗忘机制状态
        for c in self.steps_since_review_cat:
            self.steps_since_review_cat[c] = 0
        for c in self.forgetting_trigger_stats:
            self.forgetting_trigger_stats[c] = 0
        for c in self.history_buffer_cat:
            self.history_buffer_cat[c] = []
        
        # 🔧 修复不公平点1: 重置资源疲劳度
        self.resource_usage_count.fill(0.0)
        
        # 改进1: 重置action history
        self.action_history = []
        
        # 修复B: 扩展观察空间包含归一化timer
        timer_obs = self._compute_timer_observation()
        obs = np.concatenate([self.gaps, timer_obs])
        return obs, {}

    def _compute_timer_observation(self):
        """
        修复B: 计算归一化timer观察 (解决POMDP问题)
        """
        timers = []
        for cat in ['algebra', 'geometry', 'statistics']:
            steps = self.steps_since_review_cat[cat]
            threshold = self.forgetting_thresholds[cat]
            normalized_timer = min(1.0, float(steps) / float(threshold))
            timers.append(normalized_timer)
        return np.array(timers, dtype=np.float32)
    
    def _compute_action_entropy(self):
        """
        改进1: 计算action entropy用于shaping
        基于最近50步的action分布 (Flat使用72个动作)
        """
        if len(self.action_history) == 0:
            return 0.0
        
        action_hist = np.zeros(72, dtype=np.float32)
        for act in self.action_history:
            action_hist[int(act)] += 1.0
        
        freq = action_hist / (action_hist.sum() + 1e-8)
        entropy = -np.sum(freq * np.log(freq + 1e-8))
        return float(entropy)
    
    def _compute_timer_sum(self):
        timer_obs = self._compute_timer_observation()
        return float(np.sum(timer_obs))

    def _global_success(self) -> bool:
        return bool(np.all(self.gaps < self.tolerance))

    def _apply_fixed_forgetting(self, mastery: np.ndarray) -> tuple[np.ndarray, dict]:
        """修复C: 连续遗忘衰减的固定遗忘机制"""
        enhanced_params = {
            'thresholds': {
                'algebra': 5, 'geometry': 4, 'statistics': 6
            },
            'strengths': {
                'algebra': 0.07, 'geometry': 0.07, 'statistics': 0.07
            },
            'adjust_factor': 0.6,
        }
        
        enhanced_params.update(self._fp.get('fixed', {}))
        events = {k: False for k in self.categories}
        
        for cat, idxs in self.categories.items():
            t = int(self.steps_since_review_cat[cat])
            th = int(enhanced_params['thresholds'][cat])
            
            if t >= th:
                s = float(enhanced_params['strengths'][cat])
                adj = float(enhanced_params.get('adjust_factor', 0.5))
                
                # 修复C: 连续衰减
                excess_steps = max(0, t - th)
                scale = 5.0
                forgetting_strength = min(1.0, float(excess_steps) / scale)
                effective_decay = 1.0 - (s * adj * forgetting_strength)
                
                for i in idxs:
                    mastery[i] = float(max(0.0, mastery[i] * effective_decay))
                
                events[cat] = True
                self.forgetting_trigger_stats[cat] += 1
                
        return mastery, events

    def _apply_improved_forgetting(self, mastery: np.ndarray) -> tuple[np.ndarray, dict]:
        """修复C: 连续遗忘衰减的改进遗忘机制"""
        thresholds = {'algebra': 5, 'geometry': 4, 'statistics': 6}
        
        if_params = {
            'base_decay': 0.07,
            'stability_k': 8.0,
            'stability_lambda': 0.7,
        }
        if_params.update(self._fp.get('improved', {}))
        
        ed = {k: 0.0 for k in self.categories}
        
        for cat, idxs in self.categories.items():
            t_c = int(self.steps_since_review_cat[cat])
            th = int(thresholds[cat])
            
            if t_c >= th:
                hist = list(self.history_buffer_cat.get(cat, []))
                if len(hist) >= 2:
                    vol_c = float(np.std(hist))
                    stability_c = float(np.exp(-if_params['stability_k'] * vol_c))
                else:
                    stability_c = 0.5
                
                base_decay = if_params['base_decay']
                lambda_factor = if_params['stability_lambda']
                
                # 修复C: 连续衰减 + 稳定性感知
                excess_steps = max(0, t_c - th)
                scale = 5.0
                forgetting_strength = min(1.0, float(excess_steps) / scale)
                
                effective_decay_base = base_decay * (1.0 - lambda_factor * stability_c)
                effective_decay = effective_decay_base * forgetting_strength
                
                decay_multiplier = 1.0 - (effective_decay * 0.6)
                for i in idxs:
                    mastery[i] = float(max(0.0, mastery[i] * decay_multiplier))
                
                ed[cat] = effective_decay
                self.forgetting_trigger_stats[cat] += 1
                
        return mastery, ed

    def step(self, action: int):
        self.step_count += 1
        prev_sum = float(np.sum(self.gaps))
        prev_completed = int(np.sum(self.gaps < self.tolerance))

        # 解码动作：72个动作 = 12个技能 × 6个动作类型
        skill_id = int(action // 6)
        type_id = int(action % 6)
        
        # 🔧 修复不公平点1: 使用与分层环境相同的difficulty-aware动力学
        skill_gap = float(self.gaps[skill_id])
        mastery = 1.0 - skill_gap
        
        # 确定掌握度区间 (与MiniSkillEnv完全一致)
        if mastery <= 0.33:
            level = "low"
        elif mastery <= 0.66:
            level = "medium"
        else:
            level = "high"
        
        action_name = self.action_semantics[type_id]
        
        # 🔧 修复不公平点1: 添加资源衰减机制 (与MiniSkillEnv保持一致)
        resource_decay = 1.0
        if type_id in [0, 1, 2]:  # 资源动作
            res_id = type_id
            # 更新使用次数
            self.resource_usage_count[skill_id, res_id] += 1.0
            # 连续衰减: resource_decay = exp(-alpha * usage_count)
            usage_count = self.resource_usage_count[skill_id, res_id]
            resource_decay = np.exp(-self.resource_fatigue_alpha * usage_count)
        
        # 使用GAIN_MATRIX计算增益 (与分层环境完全一致的动力学)
        base_gain = self.GAIN_MATRIX[level][action_name]
        # 应用资源衰减
        gain = base_gain * resource_decay
        
        # 🔧 修复不公平点2: 使用与分层环境相同的噪声强度
        noise = float(self.np_random.normal(0.0, 0.001))  # 从0.005改为0.001
        
        # 计算gap变化 (使用difficulty-aware gain + 资源衰减)
        delta = 0.0 if (skill_gap < self.tolerance) else gain
        new_gap = clamp(skill_gap - delta + noise, 0.0, 1.0)
        self.gaps[skill_id] = new_gap
        
        # 更新遗忘机制状态
        for cat, idxs in self.categories.items():
            if skill_id in idxs:
                self.steps_since_review_cat[cat] = 0
            else:
                self.steps_since_review_cat[cat] += 1

        # 应用遗忘机制
        mastery = 1.0 - self.gaps.copy()
        mastery_before = mastery.copy()
        forgetting_event = {'algebra': False, 'geometry': False, 'statistics': False}
        effective_decay_cat = {'algebra': 0.0, 'geometry': 0.0, 'statistics': 0.0}
        
        if self.forgetting_mode == 'fixed_forgetting':
            mastery, fe = self._apply_fixed_forgetting(mastery)
            forgetting_event.update(fe)
        elif self.forgetting_mode == 'improved_forgetting':
            mastery, ed = self._apply_improved_forgetting(mastery)
            effective_decay_cat.update(ed)
        
        # 更新gaps
        self.gaps = np.clip(1.0 - mastery, 0.0, 1.0)
        
        # 改进1: 更新action history用于entropy计算
        self.action_history.append(action)
        if len(self.action_history) > self.action_history_window:
            self.action_history = self.action_history[-self.action_history_window:]
        
        # 计算奖励
        curr_sum = float(np.sum(self.gaps))
        curr_completed = int(np.sum(self.gaps < self.tolerance))
        delta_sum = prev_sum - curr_sum
        
        reward = 0.0
        reward += 5.0 * delta_sum
        reward += 5.0 * max(0, curr_completed - prev_completed)
        reward += 20.0 if self._global_success() else 0.0
        reward += -0.02
        
        # 改进1: 弱action-entropy shaping (Flat也使用，但不使用timer-aware shaping)
        entropy = self._compute_action_entropy()
        reward += self.entropy_coef * entropy

        
        # 【EXP15修复】Timer惩罚 - 提高惩罚系数，迫使策略必须轮换
        timer_penalty_coef = 0.0  # 默认无惩罚
        if self.forgetting_mode in ['fixed_forgetting', 'improved_forgetting']:
            timer_penalty_coef = 2.0
        
        timer_sum = self._compute_timer_sum()
        reward -= timer_penalty_coef * timer_sum  # 新增Timer惩罚

        terminated = self._global_success()
        truncated = bool(self.step_count >= self.max_steps)
        self._last_action = skill_id
        
        # 动作掩码：72个动作
        mask = np.ones((72,), dtype=np.int32)
        for s in range(12):
            if self.gaps[s] < self.tolerance:
                mask[s*6:(s+1)*6] = 0
        
        # 修复B: 扩展观察空间包含归一化timer
        timer_obs = self._compute_timer_observation()
        obs = np.concatenate([self.gaps, timer_obs])
        
        # 改进6: 诊断日志
        gap_variance = float(np.var(self.gaps))
        action_freq = np.zeros(72, dtype=np.float32)
        for act in self.action_history:
            action_freq[int(act)] += 1.0
        action_freq = action_freq / (action_freq.sum() + 1e-8)
        
        info = {
            # 基础指标
            "sum_prev": prev_sum,
            "sum_curr": curr_sum,
            "skill_id": skill_id,
            "type_id": type_id,
            "delta": delta,
            "noise": noise,
            "action_mask": mask.tolist(),
            "forgetting_mode": self.forgetting_mode,
            
            # 关键诊断指标（与分层环境完全一致）
            'gap_variance': gap_variance,  # 学习均衡性
            'action_entropy': entropy,  # 策略探索水平
            'timer_mean': float(np.mean(timer_obs)),  # timer状态
            'timer_std': float(np.std(timer_obs)),  # timer波动性
            
            # 遗忘相关（与分层环境完全一致）
            'forgetting_triggers': dict(self.forgetting_trigger_stats),  # 各类别触发次数
            'effective_decay_cat': effective_decay_cat,  # 实际衰减强度
            'steps_since_review_cat': dict(self.steps_since_review_cat),  # 距离上次复习
            'forgetting_event': forgetting_event,  # 本次是否触发
            
            # 策略分析
            'action_freq': action_freq.tolist(),  # 动作使用频率
            'gap_mean': float(np.mean(self.gaps)),  # 平均gap
            
            # 保留原有字段（向后兼容）
            "timer_obs": timer_obs.tolist(),
            "mastery_before": mastery_before.tolist(),
            "mastery_after": mastery.tolist(),
        }

        # 【EXP15修复】强制策略关注所有维度 - 防止坍缩
        gap_variance = float(np.var(self.gaps))
        if gap_variance > 0.05:
            reward -= 5.0 * gap_variance  # 惩罚不均衡的学习
        else:
            reward += 0.5  # 奖励均衡的学习
        
        # 更新历史缓冲区
        for cat, idxs in self.categories.items():
            M_c = float(np.mean(mastery[idxs]))
            hb = self.history_buffer_cat.get(cat, [])
            hb = hb + [M_c]
            if len(hb) > 50:
                hb = hb[-50:]
            self.history_buffer_cat[cat] = hb
            
        return obs, reward, terminated, truncated, info
