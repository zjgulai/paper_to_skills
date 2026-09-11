# paper-选题 参考文档

## 论文搜索脚本

### ArXiv API 使用

```python
import requests
import xml.etree.ElementTree as ET

def search_arxiv(keywords, max_results=10):
    """搜索 ArXiv 论文"""
    query = '+'.join(keywords.split())
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    response = requests.get(url)
    root = ET.fromstring(response.content)

    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        paper = {
            'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
            'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text,
            'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
            'id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
        }
        papers.append(paper)

    return papers

# 使用示例
papers = search_arxiv("uplift modeling causal inference", max_results=5)
for p in papers:
    print(f"Title: {p['title']}")
    print(f"ID: {p['id']}")
    print(f"Published: {p['published']}")
    print("---")
```

### 常用搜索关键词

#### 因果推断
- uplift modeling
- causal inference
- treatment effect estimation
- propensity score
- doubly robust

#### A/B实验
- A/B testing
- multi-armed bandit
- experiment design
- statistical power

#### 时间序列
- time series forecasting
- demand forecasting
- LSTM time series

#### 供应链
- inventory optimization
- supply chain management

#### 推荐系统
- recommendation system
- collaborative filtering
- cold start problem

#### 增长模型
- user growth
- LTV prediction
- churn prediction
