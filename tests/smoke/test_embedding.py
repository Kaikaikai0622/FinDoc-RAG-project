# test_embedding.py
# 第一次跑会下载模型，bge-m3 大约 2.2GB，需要等

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
embeddings = model.encode(["贵州茅台2023年营业收入为1505亿元"])
print(f"维度: {embeddings.shape}")  # 预期: (1, 1024)
print("Embedding 测试通过！")