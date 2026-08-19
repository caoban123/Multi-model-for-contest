from __future__ import annotations

import json
from pathlib import Path

from tools.phase8_benchmark import METRIC_KEYS, evaluate


def test_unlabelled_benchmark_reports_null_not_zero(tmp_path: Path) -> None:
    queries=tmp_path/'queries.json'; queries.write_text(json.dumps({'version':'v','split':'development','queries':[{'query_id':'q','query':'','events':[],'expected_video_id':None,'expected_event_frames':[]}]}),encoding='utf-8')
    result=evaluate(queries,tmp_path/'missing.sqlite3')
    assert result['availability']=='unavailable' and result['labelled_query_count']==0
    assert all(result['metrics'][key] is None for key in METRIC_KEYS)
    assert all(value is None for value in result['ablations'].values())


def test_phase8_config_keeps_vlm_off_and_holdout_frozen() -> None:
    root=Path(__file__).parents[1]; config=json.loads((root/'configs/phase8_trake_v1.json').read_text(encoding='utf-8'))
    assert config['p8_7_vlm_extension']['enabled'] is False
    assert config['p8_7_vlm_extension']['network_calls_allowed'] is False
    assert config['benchmark']['development_config_frozen'] is True
