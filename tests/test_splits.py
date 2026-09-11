from lightrppg.datasets import subject_split


def test_subject_split_is_numeric_and_disjoint() -> None:
    subjects = [f"subject{i}" for i in [1, 10, 2, 11, 3, 12, 4, 13, 5, 14]]
    splits = subject_split(subjects)
    assert splits["train"] == ["subject1", "subject2", "subject3", "subject4", "subject5", "subject10"]
    assert not (set(splits["train"]) & set(splits["valid"]))
    assert not (set(splits["train"]) & set(splits["test"]))


def test_seeded_random_split_is_reproducible() -> None:
    subjects = [f"subject{i}" for i in range(1, 21)]
    first = subject_split(subjects, mode="seeded_random", seed=7)
    second = subject_split(list(reversed(subjects)), mode="seeded_random", seed=7)
    different = subject_split(subjects, mode="seeded_random", seed=8)
    assert first == second
    assert first != different
