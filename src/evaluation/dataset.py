"""评估数据加载器

支持加载手工标注和合成的 QA 对，提供查询和统计功能。
"""
import json
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)


class EvalDataset:
    """评估数据集管理类

    支持加载、合并、筛选手工和合成的 QA 对。
    """

    def __init__(self):
        """初始化评估数据集"""
        self.manual: list[dict] = []
        self.synthetic: list[dict] = []

    def load_manual(self, path: str = "data/eval/manual_qa.json") -> int:
        """加载手工标注的 QA 对

        支持新旧两种格式：
        - 新格式 (v1.1+): {"_schema_version": "1.1", "questions": [...]}
        - 旧格式: [...]

        Returns:
            加载的记录数
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 新格式：包含 _schema_version 和 questions 数组
            if isinstance(data, dict) and "questions" in data:
                self.manual = data["questions"]
                logger.info(f"Loaded {len(self.manual)} manual QA pairs from {path}")
                return len(self.manual)
            # 旧格式：直接是数组
            elif isinstance(data, list):
                self.manual = data
                logger.info(f"Loaded {len(self.manual)} manual QA pairs from {path}")
                return len(self.manual)
            else:
                logger.error(f"Invalid format in {path}: expected list or dict with questions")
                return 0

        except FileNotFoundError:
            logger.warning(f"Manual QA file not found: {path}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            return 0

    def load_synthetic(self, path: str = "data/eval/synthetic_qa.json") -> int:
        """加载合成的 QA 对

        支持新旧两种格式：
        - 新格式 (v1.1+): {"_schema_version": "1.1", "questions": [...]}
        - 旧格式: [...]

        Args:
            path: JSON 文件路径

        Returns:
            加载的记录数
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 新格式：包含 _schema_version 和 questions 数组
            if isinstance(data, dict) and "questions" in data:
                self.synthetic = data["questions"]
                logger.info(f"Loaded {len(self.synthetic)} synthetic QA pairs from {path}")
                return len(self.synthetic)
            # 旧格式：直接是数组
            elif isinstance(data, list):
                self.synthetic = data
                logger.info(f"Loaded {len(self.synthetic)} synthetic QA pairs from {path}")
                return len(self.synthetic)
            else:
                logger.error(f"Invalid format in {path}: expected list or dict with questions")
                return 0

        except FileNotFoundError:
            logger.warning(f"Synthetic QA file not found: {path}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            return 0

    def get_all(self) -> list[dict]:
        """获取所有 QA 对（手工 + 合成）

        Returns:
            合并后的 QA 对列表
        """
        return self.manual + self.synthetic

    def get_manual(self) -> list[dict]:
        """获取手工标注的 QA 对

        Returns:
            手工 QA 对列表
        """
        return self.manual

    def get_synthetic(self) -> list[dict]:
        """获取合成的 QA 对

        Returns:
            合成 QA 对列表
        """
        return self.synthetic

    def get_by_scene(self, scene: str) -> list[dict]:
        """按场景类型筛选 QA 对

        Args:
            scene: 场景类型 (factual, comparison, policy_qa, extraction, out_of_scope)

        Returns:
            符合条件的 QA 对列表
        """
        all_data = self.get_all()
        return [item for item in all_data if item.get("scene") == scene]

    def get_by_difficulty(self, difficulty: str) -> list[dict]:
        """按难度筛选 QA 对

        Args:
            difficulty: 难度等级 (easy, medium, hard)

        Returns:
            符合条件的 QA 对列表
        """
        all_data = self.get_all()
        return [item for item in all_data if item.get("difficulty") == difficulty]

    def get_by_source(self, source: str) -> list[dict]:
        """按来源筛选 QA 对

        Args:
            source: 来源 (manual / synthetic)

        Returns:
            符合条件的 QA 对列表
        """
        if source == "manual":
            return self.manual
        elif source == "synthetic":
            return self.synthetic
        else:
            return []

    def get_by_file(self, source_file: str) -> list[dict]:
        """按源文件筛选 QA 对

        支持 source_file (旧格式字符串) 和 source_files (新格式数组)

        Args:
            source_file: 文件名关键词

        Returns:
            符合条件的 QA 对列表
        """
        all_data = self.get_all()
        return [
            item for item in all_data
            if source_file in item.get("source_file", "") or 
               source_file in item.get("source_files", [])
        ]

    def summary(self) -> dict:
        """返回数据集统计信息

        Returns:
            统计信息字典
        """
        all_data = self.get_all()

        # 统计场景分布
        scene_counter = Counter(item.get("scene", "unknown") for item in all_data)

        # 统计难度分布
        difficulty_counter = Counter(item.get("difficulty", "unknown") for item in all_data)

        # 统计文件分布 (支持 source_file 和 source_files)
        file_sources = []
        for item in all_data:
            if "source_files" in item:
                file_sources.extend(item.get("source_files", []))
            elif "source_file" in item:
                file_sources.append(item.get("source_file"))
        file_counter = Counter(file_sources)

        # 统计来源分布
        source_counter = {
            "manual": len(self.manual),
            "synthetic": len(self.synthetic),
        }

        return {
            "total": len(all_data),
            "manual_count": len(self.manual),
            "synthetic_count": len(self.synthetic),
            "scene_distribution": dict(scene_counter),
            "difficulty_distribution": dict(difficulty_counter),
            "file_distribution": dict(file_counter),
            "source_distribution": source_counter,
        }

    def print_summary(self) -> None:
        """打印数据集摘要"""
        summary = self.summary()

        print("=" * 60)
        print("Evaluation Dataset Summary")
        print("=" * 60)
        print(f"Total QA pairs: {summary['total']}")
        print(f"  - Manual: {summary['manual_count']}")
        print(f"  - Synthetic: {summary['synthetic_count']}")
        print()

        print("Scene Distribution:")
        for scene, count in summary["scene_distribution"].items():
            print(f"  - {scene}: {count}")
        print()

        print("Difficulty Distribution:")
        for difficulty, count in summary["difficulty_distribution"].items():
            print(f"  - {difficulty}: {count}")
        print()

        print("File Distribution:")
        for file, count in summary["file_distribution"].items():
            print(f"  - {file}: {count}")
        print("=" * 60)

    def filter(
        self,
        scene: Optional[str] = None,
        difficulty: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[dict]:
        """多条件筛选 QA 对

        Args:
            scene: 场景类型
            difficulty: 难度等级
            source: 来源 (manual / synthetic)

        Returns:
            符合条件的 QA 对列表
        """
        all_data = self.get_all()

        if scene:
            all_data = [item for item in all_data if item.get("scene") == scene]

        if difficulty:
            all_data = [item for item in all_data if item.get("difficulty") == difficulty]

        if source:
            if source == "manual":
                all_data = [item for item in all_data if item in self.manual]
            elif source == "synthetic":
                all_data = [item for item in all_data if item in self.synthetic]

        return all_data


def load_dataset(
    manual_path: str = "data/eval/manual_qa.json",
    synthetic_path: str = "data/eval/synthetic_qa.json",
) -> EvalDataset:
    """便捷函数：加载完整评估数据集

    Args:
        manual_path: 手工 QA 对文件路径
        synthetic_path: 合成 QA 对文件路径

    Returns:
        加载好的 EvalDataset 实例
    """
    dataset = EvalDataset()
    dataset.load_manual(manual_path)
    dataset.load_synthetic(synthetic_path)
    return dataset


if __name__ == "__main__":
    # 测试
    dataset = load_dataset()
    dataset.print_summary()
