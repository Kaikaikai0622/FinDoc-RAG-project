"""导出 chunks 辅助脚本

帮助用户浏览已入库的 chunk，用于辅助手工标注。

用法:
    # 按文件导出 chunk 列表
    python scripts/export_chunks.py --file "指南针" --output data/eval/chunks_指南针.txt
    python scripts/export_chunks.py --file "芯导科技" --output data/eval/chunks_芯导科技.txt
    python scripts/export_chunks.py --file "陕国投A" --output data/eval/chunks_陕国投A.txt

    # 导出全部
    python scripts/export_chunks.py --all --output data/eval/all_chunks.txt
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.doc_store import DocStore


def get_all_source_files() -> list[str]:
    """获取所有已存储的文件名（用于调试）"""
    doc_store = DocStore()
    chunks = doc_store.get_all_chunks()
    files = list(set(c["metadata"]["source_file"] for c in chunks))
    return files


def export_chunks(
    output_path: str,
    file_filter: str | None = None,
    export_all: bool = False,
) -> None:
    """导出 chunks 到文本文件

    Args:
        output_path: 输出文件路径
        file_filter: 文件名过滤关键词
        export_all: 是否导出全部 chunks
    """
    doc_store = DocStore()

    if export_all:
        chunks = doc_store.get_all_chunks()
        title = f"全部 Chunks (共 {len(chunks)} 条)"
    elif file_filter:
        all_chunks = doc_store.get_all_chunks()

        # 尝试多种匹配方式
        chunks = []
        matched_file = None

        # 1. 精确匹配（处理编码问题，尝试多种编码）
        for chunk in all_chunks:
            source_file = chunk["metadata"]["source_file"]
            # 尝试直接匹配
            if file_filter in source_file:
                chunks.append(chunk)
                if not matched_file:
                    matched_file = source_file
            else:
                # 尝试用编码转换来匹配
                try:
                    # 尝试将 file_filter 编码为 bytes 再用不同编码解码
                    file_filter_bytes = file_filter.encode('gbk')
                    for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                        try:
                            decoded = file_filter_bytes.decode(enc)
                            if decoded in source_file:
                                chunks.append(chunk)
                                if not matched_file:
                                    matched_file = source_file
                                break
                        except UnicodeDecodeError:
                            continue
                except Exception:
                    pass

        if matched_file:
            title = f"Chunks - {matched_file} (total: {len(chunks)})"
        else:
            # 显示所有可用文件供参考
            all_files = get_all_source_files()
            title = f"No match for '{file_filter}', total: {len(chunks)}"
            print(f"Warning: No file matched '{file_filter}'")
            print("Files in database:")
            for f in all_files:
                print(f"     {repr(f)}")
    else:
        raise ValueError("必须指定 --file 或 --all")

    # 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(title.center(80) + "\n")
        f.write("=" * 80 + "\n\n")

        current_file = None
        for i, chunk in enumerate(chunks, 1):
            source_file = chunk["metadata"]["source_file"]
            page_number = chunk["metadata"]["page_number"]
            chunk_text = chunk["chunk_text"]
            chunk_index = chunk["metadata"]["chunk_index"]

            # 文件标题
            if source_file != current_file:
                current_file = source_file
                f.write(f"\n{'─' * 80}\n")
                f.write(f"FILE: {source_file}\n")
                f.write(f"{'─' * 80}\n\n")

            # Chunk 内容
            f.write(f"[{i:04d}] Page {page_number}, Chunk {chunk_index}\n")
            f.write(f"{'─' * 40}\n")

            # 截断过长的文本
            if len(chunk_text) > 2000:
                f.write(chunk_text[:2000] + "\n... [截断]\n")
            else:
                f.write(chunk_text + "\n")

            f.write("\n")

    print(f"Exported {len(chunks)} chunks to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="导出 chunks 辅助脚本 - 用于手工标注"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="按文件名关键词导出 (如: 指南针)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="导出全部 chunks",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="列出数据库中所有已存储的文件名",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径 (--all 和 --file 时需要)",
    )

    args = parser.parse_args()

    # 列出所有文件
    if args.list:
        files = get_all_source_files()
        print("Database stored file names:")
        for f in files:
            print(f"  - {f}")
        return

    if not args.output:
        print("--output is required (unless using --list)")
        return

    export_chunks(
        output_path=args.output,
        file_filter=args.file,
        export_all=args.all,
    )


if __name__ == "__main__":
    main()
