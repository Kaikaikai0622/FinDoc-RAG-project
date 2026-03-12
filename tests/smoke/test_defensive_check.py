"""验证 SyntheticTestsetGenerator 的防御性检查"""
import pandas as pd
from src.evaluation.testset_generator import SyntheticTestsetGenerator


class FakeTestsetOldCols:
    """模拟返回旧列名的假 testset (Ragas < 0.4 格式)"""
    def to_pandas(self):
        return pd.DataFrame([{"question": "q", "ground_truth": "a"}])


class FakeTestsetNewCols:
    """模拟返回新列名的假 testset (Ragas >= 0.4 格式)"""
    def to_pandas(self):
        return pd.DataFrame([{"user_input": "测试问题", "reference": "测试答案"}])


if __name__ == "__main__":
    gen = SyntheticTestsetGenerator()

    # 测试 1: 旧列名应该触发 ValueError
    print("=" * 50)
    print("测试 1: 旧列名 (question, ground_truth)")
    try:
        gen._convert_to_qa_format(FakeTestsetOldCols())
        print("FAIL: 应该抛出 ValueError 但没有")
    except ValueError as e:
        print("PASS: 防御性检查触发")
        print(f"  错误信息: {e}")
    except Exception as e:
        print("FAIL: 抛出了非预期异常:", type(e).__name__, e)

    # 测试 2: 新列名应该正常工作
    print("=" * 50)
    print("测试 2: 新列名 (user_input, reference)")
    try:
        result = gen._convert_to_qa_format(FakeTestsetNewCols())
        print("PASS: 转换成功")
        print(f"  结果: {result}")
    except Exception as e:
        print("FAIL: 转换失败:", type(e).__name__, e)

    print("=" * 50)
    print("测试完成")
