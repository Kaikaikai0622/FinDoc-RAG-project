#!/usr/bin/env python
"""文档摄取命令行工具

支持单文件和目录批量处理。
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import IngestionPipeline
from config import DATA_RAW_DIR


# 这些取值常见于“临时禁用 CUDA”，会导致 torch 看不到 GPU。
_GPU_MASK_VALUES = {"", "-1", "none", "null"}


def ensure_ingest_gpu_visibility(force_cpu: bool = False) -> None:
    """修复 ingest 进程中的 GPU 可见性。

    默认策略：优先 GPU。若环境变量误将 GPU 屏蔽，则自动清理。
    若传入 force_cpu，则显式设置为 CPU。
    """
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("[Ingest] 已启用 --cpu，强制使用 CPU")
        return

    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return

    normalized = raw_value.strip().lower()
    if normalized in _GPU_MASK_VALUES:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        logger.warning(
            "[Ingest] 检测到 CUDA_VISIBLE_DEVICES=%r 可能屏蔽 GPU，已自动清理。",
            raw_value,
        )


def find_pdf_files(path: str) -> list[str]:
    """查找指定路径下的所有 PDF 文件

    Args:
        path: 文件或目录路径

    Returns:
        PDF 文件路径列表
    """
    p = Path(path)

    if p.is_file():
        if p.suffix.lower() == ".pdf":
            return [str(p)]
        else:
            logger.warning(f"跳过非 PDF 文件: {p}")
            return []
    elif p.is_dir():
        pdf_files = sorted(p.glob("*.pdf"))
        return [str(f) for f in pdf_files]
    else:
        logger.error(f"路径不存在: {p}")
        return []


def print_result(result: dict) -> None:
    """打印处理结果

    Args:
        result: 处理结果字典
    """
    if "error" in result:
        logger.error(f"处理失败: {result['pdf_path']}, 错误: {result['error']}")
        return

    logger.info("=" * 50)
    logger.info(f"文件: {result['pdf_path']}")
    logger.info(f"  解析页数: {result['page_count']}")
    logger.info(f"  元素数量: {result['element_count']}")
    logger.info(f"  Chunk 数量: {result['chunk_count']}")
    logger.info(f"  向量维度: {result['embedding_dim']}")
    logger.info(f"  DocStore 总量: {result['doc_store_count']}")
    logger.info(f"  VectorStore 总量: {result['vector_store_count']}")
    logger.info(f"  耗时: {result['elapsed_time']:.2f} 秒")
    logger.info("=" * 50)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="文档摄取工具 - 将 PDF 文件解析并存储到向量数据库"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="指定单个 PDF 文件路径",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DATA_RAW_DIR,
        help="指定 PDF 文件目录（默认: data/raw/）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="输出详细日志",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU（默认优先 GPU）",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ensure_ingest_gpu_visibility(force_cpu=args.cpu)

    # 确定输入路径
    if args.file:
        input_path = args.file
    else:
        input_path = args.dir

    # 查找 PDF 文件
    logger.info(f"扫描路径: {input_path}")
    pdf_files = find_pdf_files(input_path)

    if not pdf_files:
        logger.warning("未找到 PDF 文件")
        return

    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")

    # 创建 Pipeline
    pipeline = IngestionPipeline()

    # 批量处理
    total_pages = 0
    total_chunks = 0
    total_errors = 0
    total_time = 0.0

    for pdf_file in pdf_files:
        logger.info(f"\n处理文件: {pdf_file}")
        try:
            result = pipeline.run(pdf_file)
            print_result(result)

            total_pages += result.get("page_count", 0)
            total_chunks += result.get("chunk_count", 0)
            total_time += result.get("elapsed_time", 0)
        except Exception as e:
            logger.exception(f"处理文件出错: {pdf_file}, 错误: {e}")
            total_errors += 1

    # 打印汇总
    logger.info("\n" + "=" * 50)
    logger.info("【汇总统计】")
    logger.info(f"  处理文件数: {len(pdf_files)}")
    logger.info(f"  失败文件数: {total_errors}")
    logger.info(f"  总解析页数: {total_pages}")
    logger.info(f"  总 Chunk 数: {total_chunks}")
    logger.info(f"  总耗时: {total_time:.2f} 秒")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
