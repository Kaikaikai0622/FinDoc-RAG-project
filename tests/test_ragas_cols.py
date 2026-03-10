from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample
from ragas.dataset_schema import SingleTurnSample

sample = TestsetSample(
    eval_sample=SingleTurnSample(user_input="测试问题", reference="测试答案"),
    synthesizer_name="test"
)
ts = Testset(samples=[sample])
df = ts.to_pandas()
print("实际列名:", list(df.columns))
assert "user_input" in df.columns, "FAIL: 缺少 user_input 列"
assert "reference"  in df.columns, "FAIL: 缺少 reference 列"
print("PASS: 列名验证通过")
