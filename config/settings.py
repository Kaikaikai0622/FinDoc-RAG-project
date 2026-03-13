"""项目配置文件 - 集中管理所有可调参数"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ========== 切块参数 ==========
CHUNK_SIZE = 256  # 每个 chunk 的字符数
CHUNK_OVERLAP = 50  # 相邻 chunk 之间的重叠字符数

# 政策/条款类段落整段合并 chunking
# 当文本元素的 section_title 或文本内容命中这些关键词时，不按固定 token 数切分，
# 而是整段合并（上限 POLICY_MAX_CHARS），确保条款内容完整
POLICY_SECTION_KEYWORDS = [
    "利润分配", "分红", "股利分配", "派息", "送股", "转增",
    "公司章程", "议事规则", "治理准则",
    "担保", "关联交易", "风险管理", "内控",
    "会计政策", "重大变更", "重大事项",
    "承诺", "限售", "解锁", "减持",
]
POLICY_MAX_CHARS = 2000  # 政策段落 chunk 上限（远小于 bge-m3 8192 token 上限）

# ========== Embedding 参数 ==========
EMBEDDING_MODEL = "BAAI/bge-m3"  # Embedding 模型名称
EMBEDDING_DIM = 1024  # Embedding 向量维度

# ========== 存储参数 ==========
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "data" / "chroma_db")  # Chroma 持久化目录
SQLITE_DB_PATH = str(PROJECT_ROOT / "data" / "doc_store.db")  # SQLite 数据库路径
DATA_RAW_DIR = str(PROJECT_ROOT / "data" / "raw")  # 原始 PDF 文件目录

# ========== 检索参数 ==========
RETRIEVAL_TOP_K = 30   # 粗检索候选数（向量检索阶段）
RERANK_TOP_K    = 7    # 精排最终输出数（Reranker 阶段，建议范围 5-10）
USE_RERANKER    = True # 是否启用 Reranker（False 退回 Baseline）
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"

# 向后兼容别名 — retrieve() 和现有测试继续可用，无需改动任何调用方
TOP_K = RETRIEVAL_TOP_K

# ========== LLM 配置 ==========
LLM_PROVIDER = "kimi"  # 可选: "qwen", "kimi"

# 通义千问 (使用 OpenAI 兼容 API)
QWEN_MODEL = "qwen3.5-397b-a17b"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Kimi（兼容 OpenAI SDK）
KIMI_MODEL = "kimi-k2-0905-preview"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"

# ========== 日志配置 ==========
LOG_LEVEL = "INFO"

# ========== 实验配置 ==========
EXPERIMENT_CHUNK_SIZES = [256, 512, 1024]
EXPERIMENT_TOP_KS = [3, 5, 10]

# ========== Query Router 配置 ==========
ENABLE_QUERY_ROUTER = True  # 总开关，False 时回退到旧逻辑
QUERY_ROUTER_ALLOW_AUTO_FILTER_FALLBACK = True  # 自动识别公司时允许回退
QUERY_ROUTER_ALLOW_EXPLICIT_FILTER_FALLBACK = False  # 显式 filter 默认不允许回退
QUERY_ROUTER_EMPTY_RESULT_THRESHOLD = 0  # 结果数 <= 此值时触发回退（0 表示空结果才回退）
QUERY_ROUTER_DEBUG = False  # 调试模式（额外日志输出）
QUERY_ROUTER_CONFIDENCE_THRESHOLD = 0.6  # 分类置信度阈值（低于此值标记为低置信度）
