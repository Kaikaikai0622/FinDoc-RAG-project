"""Embedding 服务抽象基类"""
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingService(ABC):
    """Embedding 服务的抽象基类

    所有 Embedding 实现类都必须继承此类并实现对应接口。
    这样做的好处是：后期换 embedding 模型时，只需改实现类，调用方无需改动。
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表

        Args:
            texts: 待向量化的文本列表

        Returns:
            向量列表，每个向量为 float 列表
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取 embedding 向量的维度

        Returns:
            向量维度
        """
        pass
