from kg_tsrdeblur.evaluate_pipeline import match_counts


def test_duplicate_and_wrong_class_are_false_positives():
    gt = [dict(box=[0, 0, 10, 10], label=0)]
    pred = [dict(box=[0, 0, 10, 10], label=0), dict(box=[0, 0, 10, 10], label=0),
            dict(box=[0, 0, 10, 10], label=1)]
    assert match_counts(pred, gt) == (1, 3, 1)


def test_no_predictions_preserves_misses():
    assert match_counts([], [dict(box=[0, 0, 10, 10], label=0)]) == (0, 0, 1)
