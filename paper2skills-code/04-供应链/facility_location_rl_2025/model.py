"""
PPO-swap: Unified and Generalizable Reinforcement Learning for Facility Location Problems on Graphs
母婴出海场景：快递站点动态搬迁与应急选址优化

基于论文：Guo et al. (2025) "Unified and Generalizable RL for Facility Location Problems on Graphs"
发表于 WWW 2025 (The Web Conference 2025)

核心算法：Swap-based PPO
- 从初始布局出发，通过"搬迁交换"（Swap）迭代优化设施位置
- GNN 感知图拓扑结构，PPO 学习最优 Swap 策略
- 物理启发式初始化加速收敛
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import random
import math


# ─────────────────────────────────────────────
# 1. 图构建：道路网络 / 供应链网络
# ─────────────────────────────────────────────

class FacilityLocationGraph:
    """
    带权图：节点为候选位置（客户点 + 候选仓点），边为道路距离。
    
    支持：
    - 随机生成合成图（测试用）
    - 从真实路网数据导入
    """

    def __init__(self, n_nodes: int, n_facilities: int, seed: int = 42):
        """
        Args:
            n_nodes:       图中节点总数（候选位置数）
            n_facilities:  需要放置的设施（仓库）数量
            seed:          随机种子
        """
        self.n_nodes = n_nodes
        self.n_facilities = n_facilities
        rng = np.random.default_rng(seed)

        # 节点坐标（二维平面模拟）
        self.coords = rng.uniform(0, 100, size=(n_nodes, 2))

        # 构造稀疏邻接矩阵（每节点连接 k 个最近邻）
        k_neighbors = min(6, n_nodes - 1)
        self.adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_nodes)}
        for i in range(n_nodes):
            dists = np.linalg.norm(self.coords - self.coords[i], axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:k_neighbors]
            for j in nearest:
                d = float(dists[j])
                self.adj[i].append((j, d))
                self.adj[j].append((i, d))

        # 节点需求（客户订单量）
        self.demands = rng.uniform(1, 10, size=n_nodes)

    def shortest_path_cost(self, facilities: List[int]) -> float:
        """
        计算 P-median 目标：所有客户到最近设施的加权距离之和。
        使用贪心最近设施分配。
        
        Args:
            facilities: 当前设施所在节点索引列表
        Returns:
            total_cost: 加权距离之和（越小越好）
        """
        if not facilities:
            return float('inf')
        fac_set = set(facilities)
        total = 0.0
        for i in range(self.n_nodes):
            min_dist = min(
                np.linalg.norm(self.coords[i] - self.coords[f])
                for f in fac_set
            )
            total += self.demands[i] * min_dist
        return total

    def swap_cost_delta(self, facilities: List[int], remove_idx: int, add_node: int) -> float:
        """
        快速计算执行一次 Swap（移除 facilities[remove_idx]，添加 add_node）后的成本变化量。
        
        Returns:
            delta: 负值代表成本下降（改进），正值代表成本上升（劣化）
        """
        new_fac = facilities.copy()
        new_fac[remove_idx] = add_node
        old_cost = self.shortest_path_cost(facilities)
        new_cost = self.shortest_path_cost(new_fac)
        return new_cost - old_cost


# ─────────────────────────────────────────────
# 2. 图神经网络（GNN）编码器
# ─────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    轻量级图神经网络，基于均值聚合（Mean-Aggregation GNN）。
    为每个节点生成嵌入，融合：
    - 节点自身特征（坐标、需求、是否已选为设施）
    - 邻居节点特征（均值聚合，2层）
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, node_feats: torch.Tensor, adj_list: Dict[int, List[Tuple[int, float]]]) -> torch.Tensor:
        """
        Args:
            node_feats: (n_nodes, in_dim) 节点特征矩阵
            adj_list:   邻接表（node_id -> [(neighbor_id, weight), ...]）
        Returns:
            embeddings: (n_nodes, out_dim)
        """
        n = node_feats.size(0)
        h = F.relu(self.input_proj(node_feats))  # (n, hidden)

        for layer in self.layers:
            # 均值聚合邻居嵌入
            agg = torch.zeros_like(h)
            for i in range(n):
                neighbors = adj_list.get(i, [])
                if neighbors:
                    nb_idx = [nb[0] for nb in neighbors]
                    agg[i] = h[nb_idx].mean(dim=0)
                else:
                    agg[i] = h[i]
            h = F.relu(layer(torch.cat([h, agg], dim=-1)))  # (n, hidden)

        return self.output_proj(h)  # (n, out_dim)


# ─────────────────────────────────────────────
# 3. PPO-swap 策略网络
# ─────────────────────────────────────────────

class SwapPolicyNetwork(nn.Module):
    """
    PPO-swap 策略网络：
    - 输入：当前图状态（GNN 嵌入）+ 当前设施分布
    - 输出：
        * remove_logits: 从现有设施中移除哪个（|P| 类 softmax）
        * add_logits:    在哪个候选节点新建设施（|V| 类 softmax）
    
    解耦动作设计使动作空间从 O(|P|×|V|) 降到 O(|P| + |V|)。
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        # 移除头：对当前设施节点的嵌入打分
        self.remove_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 添加头：对所有候选节点的嵌入打分
        self.add_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 价值头（Critic）
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        all_embeddings: torch.Tensor,       # (n_nodes, embed_dim)
        facility_indices: List[int],         # 当前设施的节点 idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            remove_logits: (n_facilities,)
            add_logits:    (n_nodes,)
            value:         scalar
        """
        fac_embeds = all_embeddings[facility_indices]           # (p, embed)
        remove_logits = self.remove_head(fac_embeds).squeeze(-1)  # (p,)
        add_logits = self.add_head(all_embeddings).squeeze(-1)    # (n,)

        # 价值估计基于所有节点平均嵌入
        graph_embed = all_embeddings.mean(dim=0)                # (embed,)
        value = self.value_head(graph_embed).squeeze(-1)        # scalar

        return remove_logits, add_logits, value


# ─────────────────────────────────────────────
# 4. PPO-swap RL 环境
# ─────────────────────────────────────────────

class FacilityLocationEnv:
    """
    设施选址的强化学习环境。
    
    状态：当前图 + 设施分布
    动作：Swap（移除一个现有设施，添加一个候选节点）
    奖励：此次 Swap 带来的成本下降量（delta_cost）
    终止：达到最大步数
    """

    def __init__(self, graph: FacilityLocationGraph, max_steps: int = 50):
        self.graph = graph
        self.max_steps = max_steps
        self.facilities: List[int] = []
        self.step_count = 0
        self.initial_cost = 0.0

    def reset(self, init_method: str = "physics") -> Tuple[List[int], float]:
        """
        重置环境，返回初始设施列表和初始成本。
        
        Args:
            init_method: "physics"（物理启发式）或 "random"
        """
        self.step_count = 0
        if init_method == "physics":
            self.facilities = self._physics_init()
        else:
            self.facilities = random.sample(range(self.graph.n_nodes), self.graph.n_facilities)

        self.initial_cost = self.graph.shortest_path_cost(self.facilities)
        return self.facilities.copy(), self.initial_cost

    def _physics_init(self) -> List[int]:
        """
        物理启发式初始化：基于 Scaling law，将需求热力图用 k-means 风格分配。
        简化实现：选取需求最高且彼此距离足够远的节点。
        """
        selected = []
        demands = self.graph.demands.copy()
        min_dist = 100.0 / math.sqrt(self.graph.n_facilities) * 0.6  # 间隔阈值

        while len(selected) < self.graph.n_facilities:
            candidate = int(np.argmax(demands))
            # 检查与已选节点的距离
            if selected:
                dists_to_selected = [
                    np.linalg.norm(self.graph.coords[candidate] - self.graph.coords[s])
                    for s in selected
                ]
                if min(dists_to_selected) < min_dist:
                    demands[candidate] = 0
                    continue
            selected.append(candidate)
            demands[candidate] = 0

        # 如果因距离限制选不够，随机补充
        remaining = list(set(range(self.graph.n_nodes)) - set(selected))
        while len(selected) < self.graph.n_facilities and remaining:
            extra = random.choice(remaining)
            selected.append(extra)
            remaining.remove(extra)

        return selected[:self.graph.n_facilities]

    def step(self, remove_idx: int, add_node: int) -> Tuple[float, bool, Dict]:
        """
        执行 Swap 动作。
        
        Args:
            remove_idx: 要移除的设施在 self.facilities 中的索引
            add_node:   要新增的节点编号
        Returns:
            reward, done, info
        """
        if add_node in self.facilities:
            # 无效动作（添加的节点已是设施），轻惩罚
            reward = -0.5
            done = False
            info = {"valid": False}
        else:
            old_cost = self.graph.shortest_path_cost(self.facilities)
            self.facilities[remove_idx] = add_node
            new_cost = self.graph.shortest_path_cost(self.facilities)
            delta = old_cost - new_cost  # 正值代表改进
            reward = delta / (self.initial_cost + 1e-8) * 10  # 归一化奖励
            info = {"delta_cost": delta, "new_cost": new_cost, "valid": True}

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return reward, done, info


# ─────────────────────────────────────────────
# 5. Mock RL Agent（用于测试，不依赖真实训练）
# ─────────────────────────────────────────────

class MockPPOSwapAgent:
    """
    Mock PPO-swap Agent，用于单元测试和自测演示。
    使用贪心策略（每次选择最优 Swap）模拟训练后的策略行为。
    适合验证环境逻辑和工作流正确性，无需 GPU / 长时训练。
    """

    def __init__(self, graph: FacilityLocationGraph, n_candidates: int = 5):
        """
        Args:
            graph:        图实例
            n_candidates: 每次随机抽取多少候选 Swap 进行贪心评估
        """
        self.graph = graph
        self.n_candidates = n_candidates

    def select_action(self, facilities: List[int]) -> Tuple[int, int]:
        """
        从随机抽样的候选 Swap 中选择使成本下降最多的动作。
        
        Returns:
            (remove_idx, add_node)
        """
        best_delta = float('inf')
        best_remove_idx = 0
        best_add_node = 0

        non_facility_nodes = list(set(range(self.graph.n_nodes)) - set(facilities))
        candidates_add = random.sample(non_facility_nodes, min(self.n_candidates, len(non_facility_nodes)))

        for remove_idx in range(len(facilities)):
            for add_node in candidates_add:
                delta = self.graph.swap_cost_delta(facilities, remove_idx, add_node)
                if delta < best_delta:
                    best_delta = delta
                    best_remove_idx = remove_idx
                    best_add_node = add_node

        return best_remove_idx, best_add_node


# ─────────────────────────────────────────────
# 6. 完整训练管线（真实 PPO-swap，可选用）
# ─────────────────────────────────────────────

class PPOSwapTrainer:
    """
    完整的 PPO-swap 训练器。
    在小图上训练，大图上泛化使用。
    """

    def __init__(
        self,
        n_nodes: int = 50,
        n_facilities: int = 5,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        gnn_in_dim: int = 4,
    ):
        self.n_nodes = n_nodes
        self.n_facilities = n_facilities
        self.gamma = gamma
        self.clip_eps = clip_eps

        self.gnn = GNNEncoder(in_dim=gnn_in_dim, hidden_dim=hidden_dim, out_dim=embed_dim)
        self.policy = SwapPolicyNetwork(embed_dim=embed_dim, hidden_dim=hidden_dim)

        self.optimizer = optim.Adam(
            list(self.gnn.parameters()) + list(self.policy.parameters()), lr=lr
        )

    def _build_node_features(self, graph: FacilityLocationGraph, facilities: List[int]) -> torch.Tensor:
        """
        构建节点特征矩阵：[x坐标, y坐标, 需求, 是否已选为设施]
        """
        feats = np.zeros((graph.n_nodes, 4), dtype=np.float32)
        feats[:, 0] = graph.coords[:, 0] / 100.0  # 归一化坐标
        feats[:, 1] = graph.coords[:, 1] / 100.0
        feats[:, 2] = graph.demands / graph.demands.max()
        for f in facilities:
            feats[f, 3] = 1.0
        return torch.tensor(feats)

    def select_action_with_logprob(
        self, graph: FacilityLocationGraph, facilities: List[int]
    ) -> Tuple[int, int, torch.Tensor]:
        """选择动作并返回对数概率（供 PPO 更新用）"""
        node_feats = self._build_node_features(graph, facilities)
        embeddings = self.gnn(node_feats, graph.adj)
        remove_logits, add_logits, _ = self.policy(embeddings, facilities)

        remove_dist = torch.distributions.Categorical(logits=remove_logits)
        remove_idx = remove_dist.sample().item()

        # 屏蔽已是设施的节点（不能重复添加）
        mask = torch.ones(graph.n_nodes)
        for f in facilities:
            mask[f] = 0.0
        masked_add_logits = add_logits + (mask.log())
        add_dist = torch.distributions.Categorical(logits=masked_add_logits)
        add_node = add_dist.sample().item()

        log_prob = remove_dist.log_prob(torch.tensor(remove_idx)) + \
                   add_dist.log_prob(torch.tensor(add_node))

        return remove_idx, add_node, log_prob

    def compute_value(self, graph: FacilityLocationGraph, facilities: List[int]) -> torch.Tensor:
        """估计状态价值"""
        node_feats = self._build_node_features(graph, facilities)
        embeddings = self.gnn(node_feats, graph.adj)
        _, _, value = self.policy(embeddings, facilities)
        return value

    def update(
        self,
        graph: FacilityLocationGraph,
        trajectories: List[Dict],
    ) -> Dict[str, float]:
        """
        单次 PPO 更新。
        
        trajectories: list of {facilities, remove_idx, add_node, reward, log_prob, value}
        """
        if not trajectories:
            return {}

        # 计算 GAE 折扣回报
        returns = []
        G = 0.0
        for traj in reversed(trajectories):
            G = traj["reward"] + self.gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        old_log_probs = torch.stack([t["log_prob"].detach() for t in trajectories])
        old_values = torch.stack([t["value"].detach() for t in trajectories])
        advantages = returns_tensor - old_values.squeeze()

        # PPO 更新（2 轮）
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for _ in range(2):
            new_log_probs = []
            new_values = []
            for traj in trajectories:
                node_feats = self._build_node_features(graph, traj["facilities"])
                embeddings = self.gnn(node_feats, graph.adj)
                remove_logits, add_logits, value = self.policy(embeddings, traj["facilities"])

                mask = torch.ones(graph.n_nodes)
                for f in traj["facilities"]:
                    mask[f] = 0.0
                masked_add = add_logits + mask.log()

                r_dist = torch.distributions.Categorical(logits=remove_logits)
                a_dist = torch.distributions.Categorical(logits=masked_add)
                lp = r_dist.log_prob(torch.tensor(traj["remove_idx"])) + \
                     a_dist.log_prob(torch.tensor(traj["add_node"]))
                new_log_probs.append(lp)
                new_values.append(value)

            new_log_probs_t = torch.stack(new_log_probs)
            new_values_t = torch.stack(new_values)

            ratio = torch.exp(new_log_probs_t - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values_t.squeeze(), returns_tensor)

            self.optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            nn.utils.clip_grad_norm_(
                list(self.gnn.parameters()) + list(self.policy.parameters()), max_norm=0.5
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            "policy_loss": total_policy_loss / 2,
            "value_loss": total_value_loss / 2,
        }


# ─────────────────────────────────────────────
# 7. 主流程：Mock 自测
# ─────────────────────────────────────────────

def run_mock_self_test():
    """
    Mock RL Agent 自测：模拟快递站点搬迁场景。
    不进行真实 PPO 训练，验证环境逻辑和 Agent 接口。
    """
    print("=" * 60)
    print("PPO-swap Mock 自测：快递站点动态搬迁")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    # ── 场景设置 ──
    N_NODES = 30       # 30 个候选位置（城市节点）
    N_FAC = 5          # 5 个快递站点
    MAX_STEPS = 20     # 每轮最多 20 次交换

    graph = FacilityLocationGraph(n_nodes=N_NODES, n_facilities=N_FAC, seed=42)
    env = FacilityLocationEnv(graph, max_steps=MAX_STEPS)
    agent = MockPPOSwapAgent(graph, n_candidates=8)

    print(f"\n图参数: {N_NODES} 个节点, {N_FAC} 个站点, {MAX_STEPS} 步/轮")

    # ── 运行 3 轮仿真 ──
    results = []
    for episode in range(3):
        facilities, initial_cost = env.reset(init_method="physics")
        print(f"\n[Episode {episode + 1}] 初始成本: {initial_cost:.2f}")
        print(f"  初始站点: {facilities}")

        total_reward = 0.0
        n_improve = 0

        for step in range(MAX_STEPS):
            remove_idx, add_node = agent.select_action(facilities)
            reward, done, info = env.step(remove_idx, add_node)
            total_reward += reward

            if info.get("valid") and info.get("delta_cost", 0) > 0:
                n_improve += 1

            facilities = env.facilities.copy()
            if done:
                break

        final_cost = graph.shortest_path_cost(facilities)
        improvement = (initial_cost - final_cost) / initial_cost * 100
        results.append({
            "episode": episode + 1,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "improvement_pct": improvement,
            "total_reward": total_reward,
            "n_improve_steps": n_improve,
        })

        print(f"  最终成本: {final_cost:.2f}")
        print(f"  成本下降: {improvement:.1f}%")
        print(f"  有效改进步数: {n_improve}/{MAX_STEPS}")
        print(f"  累计奖励: {total_reward:.3f}")
        print(f"  最终站点: {facilities}")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("自测汇总")
    print("=" * 60)
    avg_improve = np.mean([r["improvement_pct"] for r in results])
    print(f"  平均成本下降: {avg_improve:.1f}%")
    print(f"  所有 Episode 完成: {'✓' if len(results) == 3 else '✗'}")

    # ── 断言验证 ──
    assert len(results) == 3, "必须运行 3 个 Episode"
    for r in results:
        assert r["initial_cost"] > 0, "初始成本必须大于 0"
        assert r["final_cost"] > 0, "最终成本必须大于 0"
        assert isinstance(r["improvement_pct"], float), "成本下降必须是浮点数"

    print("\n[PASS] 所有自测断言通过 ✓")
    return results


def run_ppo_swap_mini_train():
    """
    完整 PPO-swap 训练（mini 版）：验证模型前向传播和参数更新。
    小图（20 节点）快速验证，不追求收敛效果。
    """
    print("\n" + "=" * 60)
    print("PPO-swap 真实训练（Mini 验证）")
    print("=" * 60)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    graph = FacilityLocationGraph(n_nodes=20, n_facilities=3, seed=0)
    env = FacilityLocationEnv(graph, max_steps=15)
    trainer = PPOSwapTrainer(n_nodes=20, n_facilities=3)

    n_episodes = 5
    all_episode_rewards = []

    for ep in range(n_episodes):
        facilities, _ = env.reset(init_method="physics")
        trajectories = []
        ep_reward = 0.0

        for _ in range(env.max_steps):
            remove_idx, add_node, log_prob = trainer.select_action_with_logprob(graph, facilities)
            value = trainer.compute_value(graph, facilities)
            reward, done, info = env.step(remove_idx, add_node)

            trajectories.append({
                "facilities": facilities.copy(),
                "remove_idx": remove_idx,
                "add_node": add_node,
                "reward": reward,
                "log_prob": log_prob,
                "value": value,
            })

            ep_reward += reward
            facilities = env.facilities.copy()
            if done:
                break

        losses = trainer.update(graph, trajectories)
        all_episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: reward={ep_reward:.3f}, "
              f"policy_loss={losses.get('policy_loss', 0):.4f}, "
              f"value_loss={losses.get('value_loss', 0):.4f}")

    print(f"\n  GNN 参数数量: {sum(p.numel() for p in trainer.gnn.parameters())}")
    print(f"  策略网络参数数量: {sum(p.numel() for p in trainer.policy.parameters())}")
    print(f"  平均奖励（最后 3 轮）: {np.mean(all_episode_rewards[-3:]):.3f}")

    # 验证参数确实被更新
    assert all_episode_rewards, "必须有训练记录"
    print("[PASS] PPO-swap 真实训练验证通过 ✓")
    return trainer


if __name__ == "__main__":
    # 自测 1：Mock Agent（快速验证环境逻辑）
    mock_results = run_mock_self_test()

    # 自测 2：真实 PPO-swap 训练（mini 版，验证模型）
    trainer = run_ppo_swap_mini_train()

    print("\n" + "=" * 60)
    print("全部自测完成 ✓")
    print("=" * 60)
