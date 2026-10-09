# 企业知识库问答系统（RAG 最简版）

基于大模型 + 检索增强生成（RAG）的企业文档问答 Demo。上传企业文档后，系统自动切块、检索相关片段，再调用大模型基于片段回答；文档外问题会拒答，防止大模型幻觉。

## 功能

- 支持上传 PDF / TXT / MD 企业文档
- 文档自动切块，按问题相似度检索最相关片段
- 调用 DeepSeek 大模型基于检索片段生成回答
- 文档外问题通过 Prompt 约束实现拒答
- Streamlit 网页界面，开箱即用

## 技术栈

- Python 3
- Streamlit（Web 界面）
- DeepSeek API（大模型，OpenAI 兼容接口）
- scikit-learn（TF-IDF 相似度检索）
- pypdf（PDF 解析）

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
pip install streamlit openai scikit-learn pypdf
```

### 2. 配置 API Key

去 [platform.deepseek.com](https://platform.deepseek.com) 申请 API Key，填入 `app.py`：

```python
client = OpenAI(
    api_key="你的-DeepSeek-API-Key",
    base_url="https://api.deepseek.com"
)
```

### 3. 运行

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501，上传文档即可提问。

## 工作原理

```
上传文档 → 切块 → 问题与所有片段算 TF-IDF 相似度 → 取最相关 3 段
       → 把这 3 段 + 问题一起喂给大模型 → 大模型基于片段回答
       → 片段里没有的信息，Prompt 指令其拒答
```

这是 RAG（Retrieval-Augmented Generation）的最简实现：用关键词相似度替代向量检索，无需下载 Embedding 模型，适合快速验证 RAG 思路。正式版可升级为 Embedding 模型 + 向量数据库（如 Chroma）。

## 项目结构

```
rag_project/
├── app.py                      # 主程序
├── 示例文档_员工手册.txt        # 测试用示例文档
└── README.md
```
