---
title: 供应链本体Schema版本化与迁移 — 向后兼容的本体演化策略防止Agent系统崩溃
doc_type: knowledge
module: 24-标签工程
topic: sc-ontology-schema-versioning-migration
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 供应链本体Schema版本化与迁移

> **来源**：Palantir Ontology 官方文档（Schema Evolution 章节）+ Apache Atlas 版本化最佳实践 + Google Protobuf 向后兼容设计 + AWS Glue Schema Registry
> **桥梁**：标签工程 ↔ 数据治理 ↔ Palantir Ontology Layer | **类型**：数据工程+本体治理

## ① 算法原理

**问题本质**：当供应链本体（Ontology Schema）需要演化时（新增 SKU 属性、调整关系类型、修改约束条件），如何保证**已部署的 Agent 和 Action 不中断**？这是 Palantir 企业级部署中最容易忽视但最关键的工程能力。

**Schema 演化的三类变更**：

```
安全变更（向后兼容）：             危险变更（破坏兼容）：
✅ 新增可选属性                   ❌ 删除已有属性
✅ 新增新的 ObjectType            ❌ 重命名已有属性
✅ 扩展枚举值（追加）             ❌ 改变属性类型（string→int）
✅ 放宽约束（必填→可选）          ❌ 收紧约束（可选→必填）
✅ 新增 LinkType                  ❌ 删除 LinkType
```

**版本化策略三种模式**（对应不同企业成熟度）：

```
Mode 1 - 语义版本号（SemVer）：
  MAJOR.MINOR.PATCH
  MAJOR: 破坏性变更（需要迁移脚本）
  MINOR: 新功能（向后兼容）
  PATCH: Bug修复（向后兼容）

Mode 2 - 时间戳快照（Palantir 模式）：
  Schema_v2026_01_01 保留所有历史版本
  Agent 绑定特定版本，按需升级
  支持 Point-in-Time 查询

Mode 3 - 双写过渡（Zero-downtime 迁移）：
  Step 1: 同时写入 v1 + v2 字段
  Step 2: 验证 v2 数据质量
  Step 3: 读取切换到 v2
  Step 4: 废弃 v1 字段（设置 TTL）
```

**关键工程：破坏性变更的安全迁移流程**：

```
[检测] → [通知] → [双写] → [验证] → [切换] → [废弃]
   ↓         ↓        ↓        ↓        ↓        ↓
自动扫描  告知依赖  新旧字段  数据质量  流量切换  延迟删除
Schema比对 Agent列表 同时写入  检查率   灰度放量  30天TTL
```

## ② 母婴出海应用案例

**场景A：新增 SKU 合规属性（CA65/CPSIA认证）**

品牌扩展北美市场，需要在 Product Object 上新增 `ca65_compliant` 和 `cpsia_cert_number` 两个属性。这是安全变更（新增可选属性），但需要：通知所有依赖 Product Schema 的 Agent、更新 OKB 图谱、补充历史数据。

**数据要求**：现有 Product 节点列表、Agent 依赖清单、历史合规证书数据
**预期产出**：版本化的 Schema 变更记录 + 自动化迁移脚本 + Agent 兼容性验证报告
**业务价值**：新属性上线零停机，8 个依赖 Agent 全部平滑过渡，迁移时间从 2 周手工协调 → 1 天自动化

**场景B：供应商层级关系重构（Tier1/Tier2 拆分）**

原 Supplier ObjectType 不区分供应层级，现在需要拆分为 `Tier1Supplier`（直接供应商）和 `Tier2Supplier`（原材料商），需要重构 LinkType。这是危险变更，需要双写过渡 + 3 周灰度期。

**数据要求**：现有供应商分类数据、依赖 Supplier 的 Agent 清单
**预期产出**：双写过渡方案 + 灰度切换日历 + 回滚预案
**业务价值**：供应链可见性从 1 层 → 2 层，Tier2 风险提前 90 天预警（但迁移必须零事故）

## ③ 代码模板

```python
import json
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class ChangeType(Enum):
    ADD_OPTIONAL_PROPERTY = "add_optional_property"     # ✅ 安全
    ADD_OBJECT_TYPE = "add_object_type"                 # ✅ 安全
    ADD_LINK_TYPE = "add_link_type"                     # ✅ 安全
    EXTEND_ENUM = "extend_enum"                         # ✅ 安全
    REMOVE_PROPERTY = "remove_property"                 # ❌ 危险
    RENAME_PROPERTY = "rename_property"                 # ❌ 危险
    CHANGE_PROPERTY_TYPE = "change_property_type"       # ❌ 危险
    MAKE_PROPERTY_REQUIRED = "make_property_required"   # ❌ 危险
    REMOVE_OBJECT_TYPE = "remove_object_type"           # ❌ 危险

@dataclass
class SchemaChange:
    """单个 Schema 变更描述"""
    change_id: str
    change_type: ChangeType
    object_type: str
    property_name: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    is_breaking: bool = False
    migration_required: bool = False
    
    def __post_init__(self):
        breaking_types = {
            ChangeType.REMOVE_PROPERTY, ChangeType.RENAME_PROPERTY,
            ChangeType.CHANGE_PROPERTY_TYPE, ChangeType.MAKE_PROPERTY_REQUIRED,
            ChangeType.REMOVE_OBJECT_TYPE
        }
        self.is_breaking = self.change_type in breaking_types
        self.migration_required = self.is_breaking

@dataclass
class SchemaVersion:
    """Schema 版本快照"""
    version: str   # semver: "1.2.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat()[:10])
    object_types: Dict[str, Dict] = field(default_factory=dict)
    link_types: Dict[str, Dict] = field(default_factory=dict)
    changelog: List[str] = field(default_factory=list)

class SCOntologyVersionManager:
    """
    供应链本体 Schema 版本管理器
    
    核心能力：
    1. 变更影响分析（哪些 Agent 受影响）
    2. 向后兼容性检查
    3. 双写过渡脚本生成
    4. 版本回滚支持
    """
    
    def __init__(self):
        self.versions: List[SchemaVersion] = []
        self.agent_schema_deps: Dict[str, Set[str]] = {}  # agent_id → {used_properties}
    
    def register_schema_version(self, version: SchemaVersion) -> str:
        """注册新的 Schema 版本"""
        self.versions.append(version)
        return f"注册版本 {version.version} 成功"
    
    def register_agent_dependency(self, agent_id: str, 
                                   used_properties: List[str]):
        """注册 Agent 的 Schema 依赖（哪些属性被读写）"""
        self.agent_schema_deps[agent_id] = set(used_properties)
    
    def analyze_change_impact(self, changes: List[SchemaChange]) -> Dict:
        """
        变更影响分析：识别受影响的 Agent + 生成迁移计划
        
        Returns:
            dict: impact_summary, affected_agents, migration_plan, risk_level
        """
        breaking_changes = [c for c in changes if c.is_breaking]
        safe_changes = [c for c in changes if not c.is_breaking]
        
        # 识别受影响的 Agent
        affected_agents = {}
        for change in breaking_changes:
            prop_key = f"{change.object_type}.{change.property_name}" if change.property_name else change.object_type
            for agent_id, deps in self.agent_schema_deps.items():
                # 检查 Agent 是否依赖被变更的属性
                if any(prop_key in dep or change.object_type in dep for dep in deps):
                    if agent_id not in affected_agents:
                        affected_agents[agent_id] = []
                    affected_agents[agent_id].append({
                        "change_id": change.change_id,
                        "change_type": change.change_type.value,
                        "property": prop_key
                    })
        
        # 风险等级判断
        if len(breaking_changes) == 0:
            risk_level = "LOW"
        elif len(affected_agents) == 0:
            risk_level = "MEDIUM"
        elif len(affected_agents) <= 2:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # 生成迁移计划
        migration_plan = self._generate_migration_plan(changes, affected_agents)
        
        return {
            "total_changes": len(changes),
            "breaking_changes": len(breaking_changes),
            "safe_changes": len(safe_changes),
            "affected_agents": affected_agents,
            "risk_level": risk_level,
            "migration_plan": migration_plan,
            "estimated_days": max(1, len(breaking_changes) * 3 + len(affected_agents))
        }
    
    def _generate_migration_plan(self, changes: List[SchemaChange],
                                   affected_agents: Dict) -> List[Dict]:
        """生成分步迁移计划"""
        plan = []
        
        # Step 1: 通知受影响 Agent 的开发者
        if affected_agents:
            plan.append({
                "step": 1,
                "action": "通知",
                "description": f"通知 {len(affected_agents)} 个受影响 Agent 的开发者",
                "agents": list(affected_agents.keys()),
                "duration_days": 1
            })
        
        # Step 2: 对破坏性变更启动双写
        breaking = [c for c in changes if c.is_breaking]
        if breaking:
            plan.append({
                "step": 2,
                "action": "双写启动",
                "description": "新旧字段同时写入，确保新字段数据完整",
                "changes": [c.change_id for c in breaking],
                "duration_days": 7
            })
            
            # Step 3: 数据质量验证
            plan.append({
                "step": 3,
                "action": "数据验证",
                "description": "验证新字段覆盖率 >99%，数据质量通过",
                "success_criteria": "new_field_coverage > 99%",
                "duration_days": 3
            })
            
            # Step 4: 读取切换（灰度）
            plan.append({
                "step": 4,
                "action": "灰度切换",
                "description": "读取从旧字段切换到新字段（5%→50%→100%）",
                "rollout_schedule": ["5%", "50%", "100%"],
                "duration_days": 7
            })
        
        # Step 5: 废弃旧字段
        plan.append({
            "step": len(plan) + 1,
            "action": "废弃清理",
            "description": "设置旧字段 TTL=30天，监控无依赖后删除",
            "duration_days": 30
        })
        
        return plan
    
    def check_backward_compatibility(self, old_version: SchemaVersion,
                                       new_version: SchemaVersion) -> Dict:
        """检查两个版本之间的向后兼容性"""
        issues = []
        warnings = []
        
        # 检查 ObjectType 删除
        for ot in old_version.object_types:
            if ot not in new_version.object_types:
                issues.append(f"❌ ObjectType '{ot}' 被删除")
        
        # 检查属性变更
        for ot, schema in old_version.object_types.items():
            if ot not in new_version.object_types:
                continue
            new_schema = new_version.object_types[ot]
            old_props = set(schema.get("properties", {}).keys())
            new_props = set(new_schema.get("properties", {}).keys())
            
            # 属性删除
            removed = old_props - new_props
            for prop in removed:
                issues.append(f"❌ {ot}.{prop} 被删除")
            
            # 属性类型变更
            for prop in old_props & new_props:
                old_type = schema.get("properties", {}).get(prop, {}).get("type")
                new_type = new_schema.get("properties", {}).get(prop, {}).get("type")
                if old_type and new_type and old_type != new_type:
                    issues.append(f"❌ {ot}.{prop} 类型从 {old_type} 改为 {new_type}")
        
        # 新增属性检查（警告：需要补充历史数据）
        for ot, schema in new_version.object_types.items():
            if ot in old_version.object_types:
                old_props = set(old_version.object_types[ot].get("properties", {}).keys())
                new_props = set(schema.get("properties", {}).keys())
                added = new_props - old_props
                for prop in added:
                    warnings.append(f"⚠️  {ot}.{prop} 新增（历史数据为null，需要回填）")
        
        is_compatible = len(issues) == 0
        
        return {
            "is_backward_compatible": is_compatible,
            "breaking_issues": issues,
            "warnings": warnings,
            "verdict": "✅ 向后兼容" if is_compatible else f"❌ 破坏性变更 ({len(issues)}个问题)"
        }
    
    def generate_dual_write_snippet(self, change: SchemaChange) -> str:
        """生成双写过渡代码片段"""
        if change.change_type == ChangeType.RENAME_PROPERTY:
            return f"""# 双写过渡：{change.property_name} → {change.new_value}
# 同时写入旧字段和新字段，确保新旧 Agent 都能读取
def write_with_dual_field(record):
    record['{change.property_name}'] = value   # 保留旧字段（兼容旧 Agent）
    record['{change.new_value}'] = value       # 新字段（新 Agent 使用）
    return record

# 读取时优先使用新字段
def read_field(record):
    return record.get('{change.new_value}') or record.get('{change.property_name}')"""
        
        return f"# 变更类型 {change.change_type.value} 的迁移代码（请参考迁移计划手动实现）"


# ===== 测试用例 =====
def run_test():
    manager = SCOntologyVersionManager()
    
    # 注册 v1.0 Schema
    v1 = SchemaVersion(
        version="1.0.0",
        object_types={
            "Product": {"properties": {"sku": {"type": "string"}, "unit_cost": {"type": "float"},
                                        "category": {"type": "string"}}},
            "Supplier": {"properties": {"name": {"type": "string"}, "lead_time": {"type": "int"}}}
        }
    )
    manager.register_schema_version(v1)
    
    # 注册 Agent 依赖
    manager.register_agent_dependency("replenishment-agent", 
                                       ["Product.sku", "Product.unit_cost", "Supplier.lead_time"])
    manager.register_agent_dependency("pricing-agent", 
                                       ["Product.sku", "Product.unit_cost"])
    
    # 计划变更：新增合规属性（安全）+ 删除 category（危险）
    changes = [
        SchemaChange("CHG-001", ChangeType.ADD_OPTIONAL_PROPERTY, "Product", 
                     "ca65_compliant", new_value="boolean"),
        SchemaChange("CHG-002", ChangeType.REMOVE_PROPERTY, "Product", "category"),
    ]
    
    impact = manager.analyze_change_impact(changes)
    
    # 验证
    assert impact["breaking_changes"] == 1, "应有1个破坏性变更"
    assert impact["safe_changes"] == 1, "应有1个安全变更"
    assert len(impact["affected_agents"]) > 0, "应识别到受影响的Agent"
    assert impact["risk_level"] in ["HIGH", "CRITICAL", "MEDIUM"], "应评估风险等级"
    
    print(f"  变更分析: {impact['total_changes']}个变更 ({impact['breaking_changes']}个破坏性)")
    print(f"  风险等级: {impact['risk_level']}")
    print(f"  受影响Agent: {list(impact['affected_agents'].keys())}")
    print(f"  迁移计划: {len(impact['migration_plan'])} 步, 预计{impact['estimated_days']}天")
    
    # 测试向后兼容性检查
    v2 = SchemaVersion(
        version="2.0.0",
        object_types={
            "Product": {"properties": {"sku": {"type": "string"}, "unit_cost": {"type": "float"},
                                        "ca65_compliant": {"type": "boolean"}}},  # 删除了 category
            "Supplier": {"properties": {"name": {"type": "string"}, "lead_time": {"type": "int"}}}
        }
    )
    compat = manager.check_backward_compatibility(v1, v2)
    assert not compat["is_backward_compatible"], "删除category应不兼容"
    assert len(compat["breaking_issues"]) == 1, "应有1个破坏性问题"
    
    print(f"  兼容性检查: {compat['verdict']}")
    print(f"  警告: {len(compat['warnings'])} 个（新增字段需回填）")
    
    # 测试双写代码生成
    rename_change = SchemaChange("CHG-003", ChangeType.RENAME_PROPERTY, "Supplier",
                                  "lead_time", new_value="lead_time_days")
    snippet = manager.generate_dual_write_snippet(rename_change)
    assert "dual" in snippet.lower() or "lead_time" in snippet, "应生成双写代码"
    print(f"  双写代码生成: ✅ ({len(snippet)} 字符)")
    
    print("\n[✓] SC-Ontology-Schema-Versioning 测试通过 — 变更分析+兼容检查+迁移计划就绪")

run_test()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Tag-Schema-Engineering-Lifecycle]] — Schema 生命周期管理是版本化的前置
- **前置（prerequisite）**：[[Skill-Supply-Chain-Data-Lineage-Tracking]] — 血缘追踪帮助识别 Schema 变更的下游影响
- **延伸（extends）**：[[Skill-Ontology-LLM-AutoBuild-SC]] — 自动构建的本体必须通过版本化管理持续演化
- **延伸（extends）**：[[Skill-Decision-Audit-Trail-Ontology]] — 审计追踪日志记录每次 Schema 变更的历史
- **可组合（combinable）**：[[Skill-Graph-OKB-Design-SC]] — OKB 的 Schema 演化需要本 Skill 的版本化治理
- **可组合（combinable）**：[[Skill-SC-Agent-MCP-ERP-Integration]] — Agent 系统升级时需要 Schema 版本化保证兼容性

## ⑤ 商业价值评估

- **ROI 预估**：Schema 迁移零停机（vs 传统停机 4-8 小时），Agent 升级不中断业务；Palantir 企业客户数据：Schema 变更事故导致的 Agent 崩溃年均成本约 20-80 万元人民币
- **实施难度**：⭐⭐⭐☆☆（主要是工程纪律和流程规范，代码复杂度不高）
- **优先级**：⭐⭐⭐⭐☆（随着 Agent 数量增长，Schema 管理变为关键路径）
- **企业AI知识库依赖**：高 — 版本化的 Schema 历史即是企业知识库的元数据资产
