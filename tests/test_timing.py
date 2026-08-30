from finqa_rag.common.timing import latency_summary


def test_latency_summary_reports_linear_percentiles() -> None:
    assert latency_summary([10.0, 20.0, 30.0, 40.0, 50.0]) == {
        "p50_ms": 30.0,
        "p95_ms": 48.0,
    }


def test_latency_summary_rejects_empty_samples() -> None:
    try:
        latency_summary([])
    except ValueError as error:
        assert str(error) == "latency samples must not be empty"
    else:
        raise AssertionError("empty latency samples must raise ValueError")
