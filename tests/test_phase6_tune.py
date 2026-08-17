from aic_retrieval.reranking import RerankerConfig
from tools.phase6_tune import evaluate


def test_tuner_evaluate_uses_judged_development_candidates():
    queries=[{"text":'biển hiệu chữ "Samsung"',"expected_video_ids":["V2"],"baseline_results":[{"rank":1,"video_id":"V1","fusion_score":1,"modality_ranks":{"clip":1,"ocr":None},"frames":[]},{"rank":2,"video_id":"V2","fusion_score":.99,"modality_ranks":{"clip":2,"ocr":1},"frames":[{"evidence":{"ocr":[{"matched_text":"Samsung"}],"asr":[],"metadata":None}}]}]}]
    metrics=evaluate(queries,RerankerConfig(top_n=2))
    assert metrics=={"mrr":1.0,"top1":1.0}
