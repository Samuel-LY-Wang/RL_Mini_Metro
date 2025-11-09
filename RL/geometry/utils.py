import numpy as np

from geometry.point import Point


def distance(p1: Point, p2: Point) -> float:
    return np.sqrt((p1.left - p2.left) ** 2 + (p1.top - p2.top) ** 2)


def direction(p1: Point, p2: Point) -> Point:
    diff = p2 - p1
    diff_magnitude = distance(p1, p2)
    # Guard against zero-length vectors to avoid divide-by-zero -> NaN warnings.
    # If the two points are equal (or extremely close), return a zero vector.
    eps = 1e-8
    if diff_magnitude < eps:
        return Point(0.0, 0.0)
    return Point(diff.left / diff_magnitude, diff.top / diff_magnitude)
