from detection_fusion.Distance import calculate_distance


def test_distance():
    assert calculate_distance(0, 0, 0, 0) == 0
    assert 100 < calculate_distance(0, 0, 0.001, 0) < 120
    assert calculate_distance(0, 0, 1, 1) > 100000
