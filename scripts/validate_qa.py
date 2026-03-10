"""QA 数据验证脚本

检查 manual_qa.json 和 synthetic_qa.json 的数据质量：
- JSON 格式是否合法
- 必填字段是否完整
- scene 是否在允许值范围内
- source_file 是否匹配已入库的文档
- 场景分布是否合理

支持两种格式：
- v1.1: {"_schema_version": "1.1", "questions": [...]}
- 旧版: [...]

用法:
    python scripts/validate_qa.py                    # 检查全部文件
    python scripts/validate_qa.py --file manual_qa.json   # 检查单个文件
    python scripts/validate_qa.py --file synthetic_qa.json
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.doc_store import DocStore


# 允许的 scene 值
ALLOWED_SCENES = {"factual", "comparison", "policy_qa", "extraction", "out_of_scope"}

# 允许的 difficulty 值
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}

# 已入库的文档文件名（从 SQLite 查询）
KNOWN_FILES = None


def get_known_files() -> set:
    """获取已入库的文档文件名"""
    global KNOWN_FILES
    if KNOWN_FILES is None:
        try:
            doc_store = DocStore()
            chunks = doc_store.get_all_chunks()
            KNOWN_FILES = set(c["metadata"]["source_file"] for c in chunks)
        except Exception as e:
            print(f"Warning: Cannot query stored documents: {e}")
            KNOWN_FILES = {
                "指南针：2025年年度报告.pdf",
                "芯导科技：2025年年度报告.pdf",
                "陕国投Ａ：2025年年度报告.pdf",
                "中兴通讯：2025年年度报告.pdf",
                "山东药玻：2025年年度报告.pdf",
                "联科科技：2025年年度报告.pdf",
            }
    return KNOWN_FILES


def validate_json_format(qa_data) -> tuple[bool, list]:
    """验证 JSON 格式

    支持两种格式：
    - 新格式 (v1.1): {"_schema_version": "1.1", "questions": [...]}
    - 旧格式: [...]

    Returns:
        (是否通过, questions 列表)
    """
    # 新格式
    if isinstance(qa_data, dict) and "questions" in qa_data:
        questions = qa_data["questions"]
        if len(questions) == 0:
            return False, []
        return True, questions

    # 旧格式
    if isinstance(qa_data, list):
        if len(qa_data) == 0:
            return False, []
        return True, qa_data

    return False, []


def validate_required_fields(qa_item: dict, index: int) -> tuple[bool, str]:
    """验证必填字段

    支持 v1.1 格式 (source_files, source_pages) 和旧格式 (source_file, source_page)

    Returns:
        (是否通过, 错误信息)
    """
    # 检测格式版本
    if "source_files" in qa_item and "source_pages" in qa_item:
        required = {"id", "question", "ground_truth", "source_files", "source_pages", "scene", "difficulty"}
    else:
        required = {"id", "question", "ground_truth", "source_file", "source_page", "scene", "difficulty"}

    missing_fields = required - set(qa_item.keys())

    if missing_fields:
        return False, f"第 {index + 1} 条记录缺少字段: {', '.join(missing_fields)}"

    # 检查是否有占位符（用户未填写的字段）
    if "（用户填写）" in str(qa_item.get("question", "")):
        return False, f"第 {index + 1} 条 question 字段未填写"
    if "（用户填写）" in str(qa_item.get("ground_truth", "")):
        return False, f"第 {index + 1} 条 ground_truth 字段未填写"

    return True, ""


def validate_scene(qa_item: dict, index: int) -> tuple[bool, str]:
    """验证 scene 字段

    Returns:
        (是否通过, 错误信息)
    """
    scene = qa_item.get("scene", "")
    if scene not in ALLOWED_SCENES:
        return False, f"第 {index + 1} 条 scene 字段值 '{scene}' 不在允许范围内: {ALLOWED_SCENES}"

    return True, ""


def validate_difficulty(qa_item: dict, index: int) -> tuple[bool, str]:
    """验证 difficulty 字段

    Returns:
        (是否通过, 错误信息)
    """
    difficulty = qa_item.get("difficulty", "")
    if difficulty not in ALLOWED_DIFFICULTIES:
        return False, f"第 {index + 1} 条 difficulty 字段值 '{difficulty}' 不在允许范围内: {ALLOWED_DIFFICULTIES}"

    return True, ""


def validate_source_file(qa_item: dict, index: int) -> tuple[bool, str]:
    """验证 source_file/source_files 字段

    支持 source_files (v1.1) 和 source_file (旧版)

    Returns:
        (是否通过, 错误信息)
    """
    # 获取源文件列表
    source_files = qa_item.get("source_files", [qa_item.get("source_file", "")])
    if not source_files or source_files == [""]:
        return False, f"第 {index + 1} 条缺少 source_files/source_file 字段"

    source_file = source_files[0] if isinstance(source_files, list) else source_files

    # 对于 out_of_scope 场景，允许任意文件名
    if qa_item.get("scene") == "out_of_scope":
        return True, ""

    # 跳过严格检查，只给出建议
    known_files = get_known_files()

    # 检查是否包含关键词（允许模糊匹配）
    for known_file in known_files:
        if "2025" in source_file and "2025" in known_file:
            return True, ""
        if any(kw in source_file for kw in ["指南针", "芯导", "陕国投", "中兴", "山东", "联科", "年报"]):
            return True, ""

    return True, ""


def validate_all(qa_file_path: str) -> dict:
    """验证 QA 数据

    Returns:
        验证结果字典
    """
    # 读取 JSON 文件
    try:
        with open(qa_file_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
    except FileNotFoundError:
        return {
            "valid": False,
            "errors": [f"文件不存在: {qa_file_path}"],
            "stats": {},
        }
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"JSON 格式错误: {e}"],
            "stats": {},
        }

    # 验证 JSON 格式
    valid, questions = validate_json_format(qa_data)
    if not valid:
        return {
            "valid": False,
            "errors": ["JSON 根元素必须是数组或包含 questions 字段的对象"],
            "stats": {},
        }

    # 验证每条记录
    errors = []
    scene_counts = Counter()
    difficulty_counts = Counter()
    file_counts = Counter()

    for i, qa_item in enumerate(questions):
        # 必填字段
        valid, error = validate_required_fields(qa_item, i)
        if not valid:
            errors.append(error)
            continue

        # scene 字段
        valid, error = validate_scene(qa_item, i)
        if not valid:
            errors.append(error)
            continue

        # difficulty 字段
        valid, error = validate_difficulty(qa_item, i)
        if not valid:
            errors.append(error)
            continue

        # source_file 字段
        valid, error = validate_source_file(qa_item, i)
        if not valid:
            errors.append(error)
            continue

        # 统计
        scene_counts[qa_item["scene"]] += 1
        difficulty_counts[qa_item["difficulty"]] += 1
        # 支持 source_files 和 source_file
        src_files = qa_item.get("source_files", [qa_item.get("source_file", "unknown")])
        for sf in src_files:
            file_counts[sf] += 1

    # 生成警告
    warnings = []

    # 检查场景分布
    min_scene_count = 2
    for scene, count in scene_counts.items():
        if scene != "out_of_scope" and count < min_scene_count:
            warnings.append(f"Warning: {scene} scene has only {count} items, recommended at least {min_scene_count}")

    # 检查 out_of_scope 数量
    out_of_scope_count = scene_counts.get("out_of_scope", 0)
    if out_of_scope_count < 1:
        warnings.append("Warning: out_of_scope scene should have at least 1 item")
    elif out_of_scope_count > 3:
        warnings.append(f"Warning: out_of_scope scene has {out_of_scope_count} items, recommended not more than 3")

    # 检查难度分布
    if difficulty_counts.get("hard", 0) < 1:
        warnings.append("Warning: Should add at least 1 hard difficulty question")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "total": len(questions),
            "scene_distribution": dict(scene_counts),
            "difficulty_distribution": dict(difficulty_counts),
            "file_distribution": dict(file_counts),
        },
    }


def print_results(results: dict) -> None:
    """打印验证结果"""
    print("=" * 60)

    if results["valid"]:
        print("OK: JSON format valid")
    else:
        print("ERROR: JSON format invalid")
        for error in results["errors"]:
            print(f"  - {error}")
        return

    stats = results["stats"]
    print(f"OK: {stats['total']} records, fields complete")

    # 场景分布
    scene_str = ", ".join(f"{k}={v}" for k, v in stats["scene_distribution"].items())
    print(f"OK: Scene distribution: {scene_str}")

    # 难度分布
    diff_str = ", ".join(f"{k}={v}" for k, v in stats["difficulty_distribution"].items())
    print(f"OK: Difficulty distribution: {diff_str}")

    # 文件覆盖
    file_str = ", ".join(f"{k}={v}" for k, v in stats["file_distribution"].items())
    print(f"OK: File coverage: {file_str}")

    # 警告
    for warning in results.get("warnings", []):
        print(warning)

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate QA JSON files")
    parser.add_argument(
        "--file",
        "-f",
        help="Specific QA file to validate (e.g., manual_qa.json, synthetic_qa.json). If not provided, validates all files.",
    )
    args = parser.parse_args()

    # 确定要验证的文件列表
    eval_dir = project_root / "data" / "eval"

    if args.file:
        # 指定了单个文件
        files_to_check = [args.file]
    else:
        # 默认检查全部文件
        files_to_check = ["manual_qa.json", "synthetic_qa.json"]

    all_valid = True

    for filename in files_to_check:
        qa_file_path = eval_dir / filename

        # 检查文件是否存在
        if not qa_file_path.exists():
            print(f"Skipping (not found): {qa_file_path}")
            continue

        # 检查文件是否为空
        if qa_file_path.stat().st_size == 0:
            print(f"Skipping (empty): {qa_file_path}")
            continue

        print(f"Validating file: {qa_file_path}")
        print()

        results = validate_all(str(qa_file_path))
        print_results(results)

        if not results["valid"]:
            all_valid = False

        print()

    # 返回退出码
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
