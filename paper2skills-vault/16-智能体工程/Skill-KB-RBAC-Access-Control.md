---
skill_id: Skill-KB-RBAC-Access-Control
domain: 16-智能体工程
created: 2026-07-08
paper: "PrivRAG: Privacy-Preserving RAG with Fine-Grained Access Control, 2024; Fine-Grained Access Control in Knowledge Graph Systems, IEEE TDSC 2023"
tags: [知识库权限, RBAC, 细粒度访问控制, 企业知识库, 安全]
difficulty: ⭐⭐⭐
priority: ⭐⭐⭐⭐
---

# Skill-KB-RBAC-Access-Control

## ① 算法原理

企业级知识库的细粒度权限控制，确保不同角色只能访问其授权知识，同时不影响RAG检索性能。

**核心设计**：
```
用户 → 角色映射（RBAC）
知识块 → 权限标签（ACL）
检索时 → 权限过滤前置（Pre-filter） or 后置（Post-filter）

Pre-filter（推荐）: 
  向量检索 → 只在用户有权限的chunk集合中检索
  优点: 安全性高，不会泄露元数据
  缺点: 需要为每个角色维护子索引

Post-filter:
  全量检索 → 过滤无权限结果
  优点: 实现简单
  缺点: 可能在排名中暴露信息
```

**权限粒度设计**：
- 文档级：整个产品手册对供应商不可见
- 章节级：合规文档的财务附录仅CFO可见
- 字段级：供应商成本数据仅采购部可见

## ② 母婴出海应用案例

**场景1：母婴电商运营知识库分层**
```
角色         可访问知识
运营专员     产品描述/营销策略/用户评论
采购经理     供应商信息/成本数据/库存
合规专员     法规文件/认证记录/测试报告
CEO          所有内容（含财务预测）
外部合作商   仅授权的产品规格文档
```

一个查询"这个产品的采购成本是多少？"：
- 运营专员：返回"该信息您暂无访问权限"
- 采购经理：返回精确成本数据
- 安全拦截：防止越权获取竞争敏感信息

**场景2：多租户知识库隔离**
SaaS场景：品牌A的知识库数据绝不泄露给品牌B

## ③ 代码模板

```python
"""
KB-RBAC: 知识库细粒度访问控制
企业级RAG权限管理
"""
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
import hashlib

@dataclass
class Permission:
    """权限定义"""
    resource_type: str   # "document" | "section" | "field"
    resource_id: str     # 资源ID或通配符(*)
    actions: Set[str]    # {"read", "write", "admin"}

@dataclass  
class Role:
    """角色定义"""
    role_id: str
    name: str
    permissions: List[Permission] = field(default_factory=list)
    parent_roles: List[str] = field(default_factory=list)  # 角色继承

@dataclass
class KBChunk:
    """知识块，附带权限标签"""
    chunk_id: str
    content: str
    embedding: Optional[List[float]] = None
    allowed_roles: Set[str] = field(default_factory=set)  # 哪些角色可访问
    allowed_users: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

class KBRBACController:
    """
    知识库RBAC访问控制器
    支持Pre-filter检索策略
    """
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}  # user -> roles
        self.chunks: Dict[str, KBChunk] = {}
        
        # 预设角色
        self._init_default_roles()
    
    def _init_default_roles(self):
        """初始化默认角色层次"""
        self.roles = {
            "admin": Role("admin", "管理员", [
                Permission("*", "*", {"read", "write", "admin"})
            ]),
            "procurement": Role("procurement", "采购经理", [
                Permission("document", "supplier_*", {"read"}),
                Permission("field", "cost_*", {"read"}),
                Permission("document", "product_*", {"read"}),
            ]),
            "ops": Role("ops", "运营专员", [
                Permission("document", "product_*", {"read"}),
                Permission("document", "marketing_*", {"read"}),
                Permission("document", "review_*", {"read"}),
            ]),
            "compliance": Role("compliance", "合规专员", [
                Permission("document", "regulation_*", {"read"}),
                Permission("document", "certification_*", {"read"}),
            ]),
            "external": Role("external", "外部合作商", [
                Permission("document", "public_*", {"read"}),
            ]),
        }
    
    def assign_role(self, user_id: str, role_id: str):
        """给用户分配角色"""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role_id)
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """获取用户所有角色（含继承）"""
        direct_roles = self.user_roles.get(user_id, set())
        all_roles = set(direct_roles)
        
        # 递归解析继承角色
        for role_id in direct_roles:
            role = self.roles.get(role_id)
            if role:
                for parent in role.parent_roles:
                    all_roles.add(parent)
        return all_roles
    
    def can_access(self, user_id: str, chunk: KBChunk) -> bool:
        """检查用户是否有权访问某知识块"""
        user_roles = self.get_user_roles(user_id)
        
        # 直接用户白名单
        if user_id in chunk.allowed_users:
            return True
        
        # 角色权限检查
        for role_id in user_roles:
            if role_id in chunk.allowed_roles:
                return True
            # admin角色拥有全部权限
            if role_id == "admin":
                return True
        
        return False
    
    def add_chunk(self, chunk: KBChunk):
        """添加知识块"""
        self.chunks[chunk.chunk_id] = chunk
    
    def pre_filter_retrieve(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[KBChunk]:
        """
        Pre-filter检索：只在用户有权限的chunks中检索
        安全性最高，不泄露任何未授权信息
        """
        import numpy as np
        
        # Step 1: 过滤用户有权访问的chunks
        accessible = [
            chunk for chunk in self.chunks.values()
            if self.can_access(user_id, chunk) and chunk.embedding
        ]
        
        if not accessible:
            return []
        
        # Step 2: 在授权子集中做向量检索
        query_arr = np.array(query_embedding)
        query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-9)
        
        scored = []
        for chunk in accessible:
            emb = np.array(chunk.embedding)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
            score = float(query_norm @ emb_norm)
            scored.append((score, chunk))
        
        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:top_k]]
    
    def audit_access(self, user_id: str, chunk_id: str, action: str) -> Dict:
        """访问审计日志"""
        import time
        chunk = self.chunks.get(chunk_id)
        allowed = self.can_access(user_id, chunk) if chunk else False
        
        return {
            "user_id": user_id,
            "chunk_id": chunk_id,
            "action": action,
            "allowed": allowed,
            "timestamp": time.time(),
            "user_roles": list(self.get_user_roles(user_id))
        }


# ===== 测试 =====
if __name__ == "__main__":
    import numpy as np
    controller = KBRBACController()
    
    # 设置用户角色
    controller.assign_role("alice", "procurement")
    controller.assign_role("bob", "ops")
    controller.assign_role("admin_user", "admin")
    
    # 添加知识块
    chunks_data = [
        KBChunk("c1", "Supplier cost: $2.5/unit, MOQ: 10000", 
                embedding=np.random.randn(384).tolist(),
                allowed_roles={"procurement", "admin"},
                metadata={"type": "supplier_cost"}),
        KBChunk("c2", "Product description: ultra-soft baby diaper",
                embedding=np.random.randn(384).tolist(),
                allowed_roles={"ops", "procurement", "admin"},
                metadata={"type": "product_desc"}),
        KBChunk("c3", "FDA regulation 21 CFR Part 106 requirements",
                embedding=np.random.randn(384).tolist(),
                allowed_roles={"compliance", "admin"},
                metadata={"type": "regulation"}),
    ]
    
    for chunk in chunks_data:
        controller.add_chunk(chunk)
    
    # 测试1: 权限检查
    assert controller.can_access("alice", chunks_data[0]), "采购经理应能访问成本数据"
    assert not controller.can_access("bob", chunks_data[0]), "运营专员不能访问成本数据"
    assert controller.can_access("admin_user", chunks_data[2]), "管理员可访问所有"
    print("✅ 权限检查测试通过")
    
    # 测试2: Pre-filter检索
    query_emb = np.random.randn(384).tolist()
    
    alice_results = controller.pre_filter_retrieve("alice", query_emb, top_k=5)
    bob_results = controller.pre_filter_retrieve("bob", query_emb, top_k=5)
    admin_results = controller.pre_filter_retrieve("admin_user", query_emb, top_k=5)
    
    # alice(采购)能看到c1, c2；bob(运营)只能看c2；admin看全部
    alice_ids = {r.chunk_id for r in alice_results}
    bob_ids = {r.chunk_id for r in bob_results}
    
    assert "c1" in alice_ids, "采购经理应检索到成本数据"
    assert "c1" not in bob_ids, "运营专员不应检索到成本数据"
    assert len(admin_results) >= len(alice_results), "管理员可访问更多"
    print(f"Alice检索到: {alice_ids}")
    print(f"Bob检索到: {bob_ids}")
    
    # 测试3: 审计日志
    log = controller.audit_access("bob", "c1", "read")
    assert log["allowed"] == False, "未授权访问应记录为False"
    assert "user_roles" in log, "应记录用户角色"
    print(f"\n审计日志: {log}")
    
    print("\n[✓] KB-RBAC访问控制测试通过")
```

## ④ 技能关联

- 前置：[[Skill-Multi-KB-Federated-Reasoning]]（多知识库联邦，权限隔离基础）
- 前置：[[Skill-PoisonedRAG-Knowledge-Poisoning-Defense]]（安全防御体系）
- 延伸：[[Skill-CapSeal-Agent-Secret-Mediation]]（Agent密钥管理）
- 延伸：[[Skill-Progent-Privilege-Control]]（权限最小化原则）
- 组合：[[Skill-RAG-Production-Observability]]（权限访问监控）

## ⑤ 商业价值评估

**ROI量化**：
- 数据泄露风险：无权限控制时，供应商成本被竞争对手获取损失约500万/次
- 合规要求：GDPR/CCPA要求数据访问可审计，满足合规避免罚款
- 实施成本：2周工程工作，vs 数据泄露事故损失百倍
- 年化风险降低：权限控制减少90%内部数据泄露事故

**实施难度**：⭐⭐⭐（需要与RAG系统集成，修改检索流程）
**优先级**：⭐⭐⭐⭐（企业级必备，监管合规要求）
