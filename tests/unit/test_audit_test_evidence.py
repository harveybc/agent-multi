from tools.audit_test_evidence import parse_pytest_summary, resource_guard


def test_parse_pytest_summary() -> None:
    assert parse_pytest_summary("18 passed, 2 skipped in 1.2s") == {
        "passed": 18,
        "failed": 0,
        "skipped": 2,
    }


def test_resource_guard_blocks_active_candidate_and_hot_gpu() -> None:
    reasons = resource_guard(
        {"workers": {"omega": {"owns_candidate": True}}},
        [{"utilization_pct": 80}],
        memory_available_bytes=8 * 1024**3,
        load_one=1.0,
        cpu_count=8,
    )
    assert reasons == ["local_doin_candidate_active", "gpu_utilization_guard"]


def test_resource_guard_allows_waiting_worker_with_headroom() -> None:
    reasons = resource_guard(
        {"workers": {"omega": {"owns_candidate": False}}},
        [{"utilization_pct": 10}],
        memory_available_bytes=8 * 1024**3,
        load_one=1.0,
        cpu_count=8,
    )
    assert reasons == []
