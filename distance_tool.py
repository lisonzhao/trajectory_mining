import numpy as np
from shapely.geometry import LineString
from math import radians, sin, cos, asin, sqrt


def judge_frechet(ca, i, j, p, q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = eucl_distance(p[0], q[0])
    elif i > 0 and j == 0:
        ca[i, j] = np.max(judge_frechet(ca, i - 1, 0, p, q), eucl_distance(p[i], q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = np.max(judge_frechet(ca, 0, j - 1, q, q), eucl_distance(p[0], q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = np.max(np.min(judge_frechet(ca, i - 1, j, p, q), judge_frechet(ca, i - 1, j - 1, p, q),
                                 judge_frechet(ca, i, j - 1, p, q)), eucl_distance(p[i], q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


def frechet_distance(p, q):
    return judge_frechet(np.multiply(np.ones((len(p), len(q))), -1), len(p) - 1, len(q) - 1, p, q)


def eucl_distance(x, y):
    return np.linalg.norm(x - y)


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def hausdoff_distance(line1, line2):
    return LineString(line1).hausdorff_distance(LineString(line2))

