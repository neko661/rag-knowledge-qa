# -*- coding: utf-8 -*-
"""
企业知识库问答系统（RAG 最简版）
功能：上传企业文档(PDF/TXT/MD)，基于文档内容问答，文档外问题拒答。
"""
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf

# ====== 配置大模型（DeepSeek，便宜好申请） ======
# 去 platform.deepseek.com 申请 API Key，填到下面引号里
client = OpenAI(
    api_key="在这里填你的-DeepSeek-API-Key",
    base_url="https://api.deepseek.com"
)

st.set_page_config(page_title="企业知识库问答", layout="wide")
st.title("📄 企业知识库问答系统")
st.write("上传企业文档后，基于文档内容向我提问；文档里没有的问题我会拒答，不瞎编。")

def read_uploaded(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    else:
        return file.read().decode("utf-8", errors="ignore")

uploaded = st.file_uploader("上传文档（支持 PDF / TXT / MD）", type=["pdf", "txt", "md"])

if uploaded is not None:
    raw = read_uploaded(uploaded)
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 40]
    if not chunks:
        chunks = [raw[i:i+400] for i in range(0, len(raw), 400)]
    st.success(f"文档《{uploaded.name}》已加载，切成 {len(chunks)} 个片段")

    question = st.text_input("💬 输入你的问题")

    if question and chunks:
        vec = TfidfVectorizer().fit(chunks + [question])
        sims = cosine_similarity(vec.transform([question]), vec.transform(chunks))[0]
        top3 = sims.argsort()[-3:][::-1]
        context = "\n---\n".join(chunks[i] for i in top3)

        with st.expander("🔍 查看从文档里检索到的片段"):
            st.write(context)

        with st.spinner("思考中..."):
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content":
                        "你是企业知识库助手。只能根据下面【参考资料】回答用户问题；"
                        "如果资料里没有相关内容，必须直接说'根据现有资料无法回答'，严禁自己编造。\n\n"
                        f"参考资料：\n{context}"},
                    {"role": "user", "content": question}
                ],
                temperature=0.3
            )
        st.write("### 🤖 回答")
        st.write(resp.choices[0].message.content)
