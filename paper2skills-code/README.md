# paper2skills-code

将前沿学术论文转化为可落地的商业决策技能代码模板。

## 目录结构

```
├── causal_inference/    # 因果推断
│   └── uplift_model/    # Uplift Model
├── ab_testing/          # A/B实验
├── time_series/         # 时间序列预测
├── supply_chain/        # 供应链优化
├── recommendation/      # 推荐系统
└── growth_model/        # 增长模型
```

## 安装
```bash
pip install -r requirements.txt
```

## 使用
每个模块包含：
- `model.py` - 核心算法实现
- `example.py` - 使用示例
- `__init__.py` - 模块导出
