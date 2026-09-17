from drebin.evaluate import deduplicate, split


def test_deduplicate_removes_duplicate_hashes():
    records = [
        {"sha256": "hash_a", "pkg": "app1"},
        {"sha256": "hash_b", "pkg": "app2"},
        {"sha256": "hash_a", "pkg": "app1_copy"},  # Tekrarlayan örnek
    ]
    unique, duplicates = deduplicate(records)

    assert len(unique) == 2
    assert duplicates == 1
    assert [r["sha256"] for r in unique] == ["hash_a", "hash_b"]


def test_split_is_deterministic_and_respects_ratio():
    records = [{"sha256": f"hash_{i}"} for i in range(10)]

    # %30 test oranıyla bölme
    train1, test1 = split(records, test_fraction=0.30, seed=42)
    train2, test2 = split(records, test_fraction=0.30, seed=42)

    assert len(train1) == 7
    assert len(test1) == 3
    # Sabit seed aynı bölünmeyi üretmeli
    assert train1 == train2
    assert test1 == test2