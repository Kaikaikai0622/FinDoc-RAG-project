"""合成评估数据集生成器

使用自定义 5-Block Flow + 后处理检查生成 Synthetic QA，
完全基于项目现有 LLM 服务，不依赖 Ragas 的生成功能。

架构：
  Block 1 — Topic Extraction
  Block 2 — Question Generation
  Block 3 — Question Evolution（含口语歧义词检测 + 母公司/合并范围标注）
  Block 4 — Grounded Answer Generation
  Block 5 — Groundedness Filter
  Block 6 — Numeric/Unit Consistency Check（数值与单位字符串回查）

新增功能：
  - Block 3：禁止生成含口语化/歧义称谓（如"老谢"）的问题，否则标记 invalid_qa 并丢弃
  - Block 3：对未明确指明"母公司/合并"报表范围的问题，在 metadata 中记录
             report_scope="unspecified"，使其不参与泛化评测
  - Block 6：对涉及数值的 QA，在原 chunk 中做数值字符串回查；
             找不到则丢弃；同时检查单位（万元/亿元）的 pattern 一致性
"""
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.generation.llm_service import BaseLLMService, get_llm_service
from src.storage.doc_store import DocStore

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 噪声过滤正则
# ──────────────────────────────────────────────
_RE_ALL_NON_TEXT = re.compile(r"^[\d\s\W]+$")       # 全数字/标点/空白
_RE_PAGE_LINE = re.compile(r"^(第\s*\d+\s*页|\d+)$")  # 纯页码行

# ──────────────────────────────────────────────
# Block 3 辅助：口语化/歧义称谓检测
# ──────────────────────────────────────────────
# 姓氏集合：涵盖常见单字姓，刻意排除"化/时/区/间/型"等易构成行业术语的字
_COLLOQUIAL_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "谢窦章云苏潘葛奚范彭郎鲁韦昌马方俞任袁柳史唐费廉岑薛雷贺倪"
    "汤滕殷罗毕郝邬安常乐于傅皮卞齐康伍余元卜顾孟平黄穆萧尹"
)

# 固定口语称谓词（无需姓氏验证，直接命中）
_COLLOQUIAL_FIXED = frozenset(["老板", "老总", "老爷子", "大佬", "大哥", "老哥"])

# 粗匹配正则：老/小 + 任意单个汉字，或固定称谓词
# 注：后续由 _is_colloquial_name_match() 精确过滤是否为姓氏
_RE_COLLOQUIAL_NAME = re.compile(
    r"(?:老|小)([\u4e00-\u9fff])"          # 老X / 小X，X 为任意汉字
    r"|老板|老总|老爷子|大佬|大哥|老哥"    # 固定称谓词
)


def _is_colloquial_name_match(text: str) -> bool:
    """判断文本中是否含口语化/歧义人名称谓。

    两步过滤：
    1. _RE_COLLOQUIAL_NAME 粗匹配（快速排除不含相关模式的文本）
    2. 对"老X"/"小X"形式，精确验证 X 是否在姓氏集合中
       → 排除"老化"、"小时"、"老区"等行业术语
    """
    for m in _RE_COLLOQUIAL_NAME.finditer(text):
        matched = m.group(0)
        # 固定称谓词直接命中
        if matched in _COLLOQUIAL_FIXED:
            return True
        # "老X" / "小X"：验证 X 是否为姓氏
        surname = m.group(1)  # group(1) = 老/小 后面捕获的那个汉字
        if surname and surname in _COLLOQUIAL_SURNAMES:
            return True
    return False

# ──────────────────────────────────────────────
# Block 3 辅助：母公司/合并报表范围检测
# ──────────────────────────────────────────────
# 若问题中 *未* 出现以下任一关键词，则认为报表范围未指明
_RE_REPORT_SCOPE_SPECIFIED = re.compile(
    r"母公司|合并|合并报表|合并财务报表|集团|母公司报表|母公司财务报表"
)

# ──────────────────────────────────────────────
# Block 6：数值/单位一致性检查
# ──────────────────────────────────────────────
# 从文本中提取所有数值（含逗号千分位、小数点）
_RE_NUMBER = re.compile(r"\d[\d,，]*(?:\.\d+)?")

# 财务常用单位关键词
_UNIT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("万元",  re.compile(r"万\s*元")),
    ("亿元",  re.compile(r"亿\s*元")),
    ("万",    re.compile(r"(?<!\d)万(?!元)")),    # 单独"万"且后面不是"元"
    ("亿",    re.compile(r"(?<!\d)亿(?!元)")),    # 单独"亿"且后面不是"元"
    ("元",    re.compile(r"(?<!万)(?<!亿)\s*元(?!\s*[0-9])")),  # 纯"元"
    ("%",     re.compile(r"[%％]")),
    ("百分点", re.compile(r"百分点")),
]

# 是否涉及数值的简单判断（答案中含数字）
_RE_HAS_NUMBER = re.compile(r"\d")


def _extract_numbers_from_text(text: str) -> list[str]:
    """提取文本中所有数值字符串（统一化：去除千分位逗号）。"""
    raw_nums = _RE_NUMBER.findall(text)
    # 统一化：去除千分位分隔符，方便比较
    return [n.replace(",", "").replace("，", "") for n in raw_nums]


def _check_numeric_consistency(answer: str, ground_truth_context: str, chunk_text: str) -> tuple[bool, str]:
    """Block 6：数值/单位一致性检查。

    逻辑：
    1. 若 answer 中不含任何数字，跳过检查（返回 passed=True）
    2. 从 answer 中提取所有数值，逐一在 chunk_text 中做字符串回查
       - 若有任意一个数值在 chunk_text 中找不到 → discard
    3. 检查 answer 中出现的单位，若 chunk_text 中对应单位完全不存在 → discard

    Args:
        answer: Block 4 产出的答案文本
        ground_truth_context: supporting_excerpt 或完整 chunk
        chunk_text: 原始 chunk（用于数值回查的权威来源）

    Returns:
        (passed: bool, reason: str)
        passed=False 表示数值/单位回查失败，需要丢弃该 QA
    """
    # 1. 不含数字 → 跳过
    if not _RE_HAS_NUMBER.search(answer):
        return True, "no numeric content, skip check"

    # 2. 数值回查
    answer_nums = _extract_numbers_from_text(answer)
    for num in answer_nums:
        # 在原 chunk 中查找（允许带/不带千分位逗号的变体）
        # 构造带千分位逗号的变体（如 1234567 → 1,234,567）
        try:
            int_part = num.split(".")[0]
            formatted = f"{int(int_part):,}"  # 英文逗号千分位
        except ValueError:
            formatted = num
        # 任意一种形式存在即可
        raw_num_in_chunk = num in chunk_text.replace(",", "").replace("，", "")
        formatted_in_chunk = formatted in chunk_text
        original_in_chunk = num in chunk_text
        if not (raw_num_in_chunk or formatted_in_chunk or original_in_chunk):
            reason = f"数值 '{num}' 在原始 chunk 中找不到，疑似幻觉"
            logger.warning("[Block6] %s", reason)
            return False, reason

    # 3. 单位检查
    answer_units = [name for name, pat in _UNIT_PATTERNS if pat.search(answer)]

    # 预计算 chunk 中的货币信号：万元/亿元/人民币/元 任意一个存在即为"有货币上下文"
    _RE_CURRENCY_SIGNAL = re.compile(r"万\s*元|亿\s*元|人民币|(?<!万)(?<!亿)\s*元")
    chunk_has_currency_signal = bool(_RE_CURRENCY_SIGNAL.search(chunk_text))
    # 另外：chunk 中含大量数字（≥3 个独立数值）也视为有货币上下文
    chunk_number_count = len(_RE_NUMBER.findall(chunk_text))

    for unit_name, unit_pat in _UNIT_PATTERNS:
        if unit_name not in answer_units:
            continue
        if unit_pat.search(chunk_text):
            continue  # chunk 中直接找到该单位，正常通过

        # ── 对"元"做特殊处理 ──
        # 原文中未出现裸"元"，但存在以下任一情况时，不直接丢弃，交 Block 5 语义判断：
        #   a. chunk 中有"万元"或"亿元"（单位继承：答案将万元/亿元换算成元）
        #   b. chunk 中有"人民币"等货币标识
        #   c. chunk 中大量数字（≥3），"元"是默认基准单位可省略
        if unit_name == "元":
            has_larger_unit = bool(re.search(r"万\s*元|亿\s*元", chunk_text))
            if has_larger_unit or chunk_has_currency_signal or chunk_number_count >= 3:
                logger.debug(
                    "[Block6] answer 含'元'但 chunk 无裸'元'，检测到货币信号，跳过强制匹配，交 Block5"
                )
                continue  # 放行，让 Block 5 判断数值换算是否正确

        reason = f"单位 '{unit_name}' 在答案中出现，但原始 chunk 中未找到对应单位"
        logger.warning("[Block6] %s", reason)
        return False, reason

    return True, "ok"


def _detect_report_scope(question: str) -> str:
    """检测问题是否明确指定了报表范围（母公司/合并）。

    Returns:
        "specified"   — 问题中明确提到了母公司或合并相关词汇
        "unspecified" — 问题未指明报表范围，存在歧义
    """
    if _RE_REPORT_SCOPE_SPECIFIED.search(question):
        return "specified"
    return "unspecified"


def _is_noise_chunk(text: str) -> bool:
    """判断 chunk 是否为噪声（需跳过）"""
    stripped = text.strip()
    if len(stripped) < 80:
        return True
    if _RE_ALL_NON_TEXT.match(stripped):
        return True
    if _RE_PAGE_LINE.match(stripped):
        return True
    return False


# ──────────────────────────────────────────────
# Block 1 — Topic Extraction
# ──────────────────────────────────────────────

def block_topic_extraction(llm: BaseLLMService, document: str, document_outline: str) -> str:
    """从文档片段中提取核心主题词。

    Returns:
        逗号分隔的 1–3 个主题词字符串；解析失败时返回 document_outline。
    """
    system_prompt = (
        "你是一个文本分析助手。你的任务是从给定的文档片段中提取1到3个核心主题词。"
        "请以 JSON 格式返回，格式为：{\"topics\": [\"主题1\", \"主题2\"]}。"
        "不要返回任何其他内容，只返回 JSON。"
    )
    user_message = f"请从以下文档片段中提取核心主题词：\n\n{document}"

    try:
        raw = llm.chat(system_prompt, user_message)
        match = re.search(r"\{.*}", raw, re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 结构")
        data = json.loads(match.group())
        topics = data.get("topics", [])
        if topics:
            return "，".join(str(t) for t in topics[:3])
        raise ValueError("topics 为空")
    except Exception as e:
        logger.warning(f"[Block1] 解析失败，使用 document_outline 作为 fallback: {e}")
        return document_outline


# ──────────────────────────────────────────────
# Block 2 — Question Generation
# ──────────────────────────────────────────────

def block_question_generation(
    llm: BaseLLMService, document: str, document_outline: str, topic: str
) -> Optional[str]:
    """根据文档内容和主题生成一个初始问题。

    Returns:
        raw_question 字符串，解析失败返回 None（跳过该样本）。
    """
    system_prompt = (
        "你是一个问题生成专家，专注于财务文档分析。"
        "根据提供的文档片段和主题，生成一个初始问题。"
        "要求：问题必须能且仅能用该文档片段回答，不能是泛泛而谈的问题。"
        "请以 JSON 格式返回，格式为：{\"question\": \"...\"}。"
        "不要返回任何其他内容，只返回 JSON。"
    )
    user_message = (
        f"文档来源：{document_outline}\n"
        f"核心主题：{topic}\n\n"
        f"文档内容：\n{document}\n\n"
        "请根据以上内容生成一个具体的问题。"
    )

    try:
        raw = llm.chat(system_prompt, user_message)
        match = re.search(r"\{.*}", raw, re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 结构")
        data = json.loads(match.group())
        question = data.get("question", "").strip()
        if not question:
            raise ValueError("question 为空")
        return question
    except Exception as e:
        logger.warning(f"[Block2] 解析失败，跳过该样本: {e}")
        return None


# ──────────────────────────────────────────────
# Block 3 — Question Evolution
# ──────────────────────────────────────────────

def block_question_evolution(
    llm: BaseLLMService, raw_question: str
) -> tuple[str, bool, str]:
    """将初始问题改写为更自然的用户提问风格。

    改进点：
    - Prompt 明确禁止口语化/歧义称谓（如"老谢"），LLM 若仍生成此类词汇，
      后处理检测后将 is_invalid 置 True，调用方应丢弃该 QA
    - 检测问题是否明确指明了报表范围（母公司/合并），若未指明则记录 report_scope

    Returns:
        (question, is_invalid, report_scope)
        - question:     改写后的问题（或 raw_question fallback）
        - is_invalid:   True 表示该问题含口语歧义称谓，应标记为 invalid_qa 并丢弃
        - report_scope: "specified" 或 "unspecified"
    """
    system_prompt = (
        "你是一个专业的财务问题改写专家。\n"
        "将给定的问题改写为真实用户在向财务分析师或对话 AI 提问时的自然语言风格。\n"
        "严格要求：\n"
        "1. 保留原始语义，不得丢失任何关键财务信息\n"
        "2. 语气自然，不能是可以直接 Ctrl+F 搜到的逐字提问\n"
        "3. 【强制禁止】使用任何人名昵称、口语化称谓或歧义表达，"
        "例如：\"老谢\"、\"老王\"、\"小李\"、\"老板\"、\"大佬\" 等；"
        "若原问题含此类词汇，必须替换为正式的职位/公司名称\n"
        "4. 若涉及财务数据，必须保留具体数值和单位，不得用模糊表述替代\n"
        "请以 JSON 格式返回，格式为：{\"evolved_question\": \"...\"}。\n"
        "不要返回任何其他内容，只返回 JSON。"
    )
    user_message = f"请改写以下问题：\n\n{raw_question}"

    try:
        raw = llm.chat(system_prompt, user_message)
        match = re.search(r"\{.*}", raw, re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 结构")
        data = json.loads(match.group())
        evolved = data.get("evolved_question", "").strip()
        if not evolved:
            raise ValueError("evolved_question 为空")
        question = evolved
    except Exception as e:
        logger.warning(f"[Block3] 解析失败，使用 raw_question 作为 fallback: {e}")
        question = raw_question

    # ── 后处理 1：口语歧义称谓检测 ──
    is_invalid = False
    if _is_colloquial_name_match(question):
        logger.warning(
            "[Block3] 检测到口语化/歧义称谓，标记 invalid_qa: %s", question
        )
        is_invalid = True

    # ── 后处理 2：报表范围检测 ──
    report_scope = _detect_report_scope(question)
    if report_scope == "unspecified":
        logger.debug(
            "[Block3] 问题未明确指明母公司/合并范围，report_scope=unspecified: %s", question
        )

    return question, is_invalid, report_scope


# ──────────────────────────────────────────────
# Block 4 — Grounded Answer Generation
# ──────────────────────────────────────────────

def block_grounded_answer(
    llm: BaseLLMService, document: str, question: str
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """仅根据文档内容回答问题，并提取支撑原文。

    Returns:
        (answer, ground_truth_context, scene, difficulty)
        若文档无法支撑回答，answer = "INSUFFICIENT_CONTEXT"。
        解析失败返回 (None, None, None, None)（跳过该样本）。
    """
    system_prompt = (
        "你是一个严格基于文档的问答助手，专注于财务文档。"
        "请仅根据提供的文档内容回答问题，禁止引入文档以外的任何知识。"
        "如果文档内容无法支撑回答，请在 answer 字段填写字符串 INSUFFICIENT_CONTEXT（全大写，固定格式）。"
        "请同时判断问题的场景类型和难度：\n"
        "- scene: factual（事实性，如数值、名称）, comparison（比较性）, extraction（提取性）, policy_qa（政策解读）\n"
        "- difficulty: easy（简单事实）, medium（需要理解）, hard（需要综合分析）\n"
        "请以 JSON 格式返回，格式为：\n"
        "{\"answer\": \"...\", \"supporting_excerpt\": \"文档中支撑答案的原文片段\", \"scene\": \"...\", \"difficulty\": \"...\"}\n"
        "不要返回任何其他内容，只返回 JSON。"
    )
    user_message = (
        f"文档内容：\n{document}\n\n"
        f"问题：{question}"
    )

    try:
        raw = llm.chat(system_prompt, user_message)
        match = re.search(r"\{.*}", raw, re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 结构")
        data = json.loads(match.group())
        answer = data.get("answer", "").strip()
        excerpt = data.get("supporting_excerpt", "").strip()
        scene = data.get("scene", "").strip().lower()
        difficulty = data.get("difficulty", "").strip().lower()
        if not answer:
            raise ValueError("answer 为空")
        # 校验 scene 和 difficulty 的有效值
        valid_scenes = {"factual", "comparison", "extraction", "policy_qa"}
        valid_difficulties = {"easy", "medium", "hard"}
        if scene not in valid_scenes:
            scene = "factual"  # 默认值
        if difficulty not in valid_difficulties:
            difficulty = "medium"  # 默认值
        ground_truth_context = excerpt if excerpt else document
        return answer, ground_truth_context, scene, difficulty
    except Exception as e:
        logger.warning(f"[Block4] 解析失败，跳过该样本: {e}")
        return None, None, None, None


# ──────────────────────────────────────────────
# Block 5 — Groundedness Filter
# ──────────────────────────────────────────────

def block_groundedness_filter(
    llm: BaseLLMService, question: str, answer: str, ground_truth_context: str
) -> bool:
    """判断 QA pair 是否满足质量要求。

    Returns:
        True = 通过过滤，False = 不通过。
    """
    if answer == "INSUFFICIENT_CONTEXT":
        logger.debug("[Block5] answer == INSUFFICIENT_CONTEXT，直接拒绝")
        return False

    system_prompt = (
        "你是一个 QA 质量评审专家。"
        "请判断给定的 QA pair 是否同时满足以下三个条件：\n"
        "1. 答案完全基于提供的上下文（无幻觉）\n"
        "2. 问题能被该上下文唯一回答（不模糊）\n"
        "3. 答案不是 INSUFFICIENT_CONTEXT\n"
        "请以 JSON 格式返回，格式为：{\"passed\": true/false, \"reason\": \"简短理由\"}。"
        "不要返回任何其他内容，只返回 JSON。"
    )
    user_message = (
        f"上下文：\n{ground_truth_context}\n\n"
        f"问题：{question}\n\n"
        f"答案：{answer}"
    )

    try:
        raw = llm.chat(system_prompt, user_message)
        match = re.search(r"\{.*}", raw, re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 结构")
        data = json.loads(match.group())
        passed = bool(data.get("passed", False))
        reason = data.get("reason", "")
        logger.debug(f"[Block5] passed={passed}, reason={reason}")
        return passed
    except Exception as e:
        logger.warning(f"[Block5] 解析失败，默认拒绝: {e}")
        return False


# ──────────────────────────────────────────────
# 采样策略
# ──────────────────────────────────────────────

def _sample_chunks(all_chunks: list[dict], sample_size: int) -> list[dict]:
    """按文件名均匀随机采样 chunk，确保每个文件都有贡献。"""
    valid_chunks = [c for c in all_chunks if not _is_noise_chunk(c.get("chunk_text", ""))]
    logger.info(f"有效 chunk 数（过滤噪声后）: {len(valid_chunks)}")

    if len(valid_chunks) == 0:
        return []

    if len(valid_chunks) <= sample_size:
        logger.info(f"有效 chunk 数不足 {sample_size}，返回全部 {len(valid_chunks)} 个")
        return valid_chunks

    by_file: dict[str, list] = defaultdict(list)
    for chunk in valid_chunks:
        source_file = chunk.get("metadata", {}).get("source_file", "unknown")
        by_file[source_file].append(chunk)

    files = list(by_file.keys())
    num_files = len(files)
    logger.info(f"共 {num_files} 个文件参与采样")

    base_quota = sample_size // num_files
    remainder = sample_size % num_files
    extra_files = set(random.sample(files, min(remainder, num_files)))

    sampled: list[dict] = []
    for f in files:
        quota = base_quota + (1 if f in extra_files else 0)
        pool = by_file[f]
        quota = min(quota, len(pool))
        sampled.extend(random.sample(pool, quota))

    if len(sampled) < sample_size:
        sampled_ids = set(id(c) for c in sampled)
        remaining = [c for c in valid_chunks if id(c) not in sampled_ids]
        extra_needed = sample_size - len(sampled)
        if remaining:
            sampled.extend(random.sample(remaining, min(extra_needed, len(remaining))))

    logger.info(f"实际采样 {len(sampled)} 个 chunk")
    return sampled


# ──────────────────────────────────────────────
# 主函数（对外暴露）
# ──────────────────────────────────────────────

def generate_synthetic_qa(
    llm_service: BaseLLMService,
    num_questions: int = 50,
    output_path: str = "data/eval/synthetic_qa.json",
) -> list[dict]:
    """使用 5-Block Flow 生成 Synthetic QA 并写入 JSON 文件。

    Args:
        llm_service: 已初始化的 LLM 服务实例（BaseLLMService 子类）
        num_questions: 目标生成题目数（通过 Groundedness Filter 的数量）
        output_path: 输出 JSON 文件路径

    Returns:
        通过过滤的 QA pair 列表（Schema v1.1 格式）
    """
    doc_store = DocStore()
    all_chunks = doc_store.get_all_chunks()
    logger.info(f"DocStore 共 {len(all_chunks)} 个 chunk")

    sample_size = min(num_questions * 2, 200)
    chunks = _sample_chunks(all_chunks, sample_size)

    if not chunks:
        logger.warning("没有可用的 chunk，无法生成 QA")
        return []

    qa_pairs: list[dict] = []
    passed_count = 0

    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text", "")
        metadata = chunk.get("metadata", {})
        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", 0)
        document_outline = f"{source_file} 第{page_number}页"

        logger.info(
            f"[{idx+1}/{len(chunks)}] 处理 chunk: {document_outline} (len={len(chunk_text)})"
        )

        # Block 1
        topic = block_topic_extraction(llm_service, chunk_text, document_outline)
        logger.debug(f"  Topic: {topic}")

        # Block 2
        raw_question = block_question_generation(
            llm_service, chunk_text, document_outline, topic
        )
        if raw_question is None:
            logger.info("  [跳过] Block2 返回 None")
            continue

        # Block 3 — 返回 (question, is_invalid, report_scope)
        question, is_invalid, report_scope = block_question_evolution(llm_service, raw_question)
        if is_invalid:
            logger.info("  [丢弃] Block3 检测到口语歧义称谓，标记 invalid_qa")
            continue
        logger.debug(f"  Question: {question}, report_scope={report_scope}")

        # Block 4
        result = block_grounded_answer(llm_service, chunk_text, question)
        answer, ground_truth_context, scene, difficulty = result
        if answer is None:
            logger.info("  [跳过] Block4 返回 None")
            continue

        # Block 6 — 数值/单位一致性检查（纯规则，零 LLM 成本，提前拦截数值幻觉）
        # 故意置于 Block 5 之前：先用确定性规则过滤，节省 Block 5 的 LLM 调用
        numeric_ok, numeric_reason = _check_numeric_consistency(
            answer, ground_truth_context, chunk_text
        )
        if not numeric_ok:
            logger.info("  [丢弃] Block6 数值/单位回查失败: %s", numeric_reason)
            continue

        # Block 5 — Groundedness Filter（LLM 语义判断，最后一道关卡）
        passed = block_groundedness_filter(
            llm_service, question, answer, ground_truth_context
        )
        if not passed:
            logger.info("  [过滤] 未通过 Groundedness Filter")
            continue

        passed_count += 1
        qa_id = f"s{passed_count:03d}"

        # 构建 qa_metadata：report_scope 若为 unspecified 则记录，供下游过滤使用
        qa_metadata: dict = {}
        if report_scope == "unspecified":
            qa_metadata["report_scope"] = "unspecified"
            qa_metadata["generalization_excluded"] = True

        qa_entry: dict = {
            "id": qa_id,
            "question": question,
            "ground_truth": answer,
            "source_files": [source_file],
            "source_pages": [page_number],
            "scene": scene,
            "difficulty": difficulty,
        }
        if qa_metadata:
            qa_entry["metadata"] = qa_metadata

        qa_pairs.append(qa_entry)
        logger.info(
            "  [通过] id=%s, report_scope=%s, 累计通过=%d",
            qa_id, report_scope, passed_count,
        )

    logger.info(
        f"生成完成：处理 {len(chunks)} 个 chunk，通过 {len(qa_pairs)} 条 QA pair"
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "_schema_version": "1.1",
        "questions": qa_pairs,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"已写入 {output_path}")

    return qa_pairs


# ──────────────────────────────────────────────
# 向后兼容包装器（SyntheticTestsetGenerator）
# ──────────────────────────────────────────────

class SyntheticTestsetGenerator:
    """向后兼容包装器，内部委托给 generate_synthetic_qa。"""

    def __init__(self, doc_store: Optional[DocStore] = None):
        self._doc_store = doc_store

    def generate(self, num_questions: int = 50) -> list[dict]:
        llm = get_llm_service()
        return generate_synthetic_qa(
            llm_service=llm,
            num_questions=num_questions,
            output_path="data/eval/synthetic_qa.json",
        )

    def save(self, testset: list[dict], path: str = "data/eval/synthetic_qa.json") -> None:
        """generate_synthetic_qa 已自动写入文件，此方法保留接口兼容性。"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "_schema_version": "1.1",
            "questions": testset,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存 {len(testset)} 条合成 QA 对到: {path}")
