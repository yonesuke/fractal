import numpy as np

def barnsley_fern(n_points: int, seed: int = None) -> np.ndarray:
    """Generates a Barnsley fern
    
    Parameters
    ----------
    n_points : int
        Number of points to generate
    seed : int, optional
        Seed for the random number generator
        
    Returns
    -------
    points : ndarray
        Array of points generated
    """
    if seed is not None:
        np.random.seed(seed)
    points = np.zeros((2, n_points), dtype=np.float64)
    for i in range(1, n_points):
        r = np.random.random()
        if r <= 0.01:
            points[0, i] = 0.0
            points[1, i] = 0.16 * points[1, i - 1]
        elif r <= 0.86:
            points[0, i] = 0.85 * points[0, i - 1] + 0.04 * points[1, i - 1]
            points[1, i] = -0.04 * points[0, i - 1] + 0.85 * points[1, i - 1] + 1.6
        elif r <= 0.93:
            points[0, i] = 0.2 * points[0, i - 1] - 0.26 * points[1, i - 1]
            points[1, i] = 0.23 * points[0, i - 1] + 0.22 * points[1, i - 1] + 1.6
        else:
            points[0, i] = -0.15 * points[0, i - 1] + 0.28 * points[1, i - 1]
            points[1, i] = 0.26 * points[0, i - 1] + 0.24 * points[1, i - 1] + 0.44
    return points

# rotate 90 degrees clockwise
def _rotate90_hilbert(points, k=1):
    # points is a 2xN array
    assert points.shape[0] == 2
    # k mod 4
    k = k % 4
    points_rot = np.array([-points[1], points[0]])
    if k == 1:
        return points_rot
    else:
        return _rotate90_hilbert(points_rot, k-1)

def _hilbert_top_left(points):
    # points is a 2xN array
    points_top_left = _rotate90_hilbert(points) # # rotate 90 degrees anti-clockwise
    points_top_left[0, :] += points[0].max() # add max y + 1 to x coordinates
    points_top_left[1, :] += points[1].max() + 1 # add max x to y coordinates
    points_top_left = np.fliplr(points_top_left) # # flip left-right
    return points_top_left

def _hilbert_bottom_left(points):
    # points is a 2xN array
    return points.copy()

def _hilbert_bottom_right(points):
    # points is a 2xN array
    points_bottom_right = points.copy() # no rotation
    points_bottom_right[0, :] += points[0].max() + 1 # add max x + 1 to x coordinates
    return points_bottom_right

def _hilbert_top_right(points):
    # points is a 2xN array
    points_top_right = _rotate90_hilbert(points, 3) # rotate 270 degrees anti-clockwise
    points_top_right[0, :] += points[0].max() + 1 # add max x + 1 to x coordinates
    points_top_right[1, :] += 2 * points[1].max() + 1 # add 2 * (max y) + 1 to y coordinates
    points_top_right = np.fliplr(points_top_right)
    return points_top_right

def hilbert_curve(n: int) -> np.ndarray:
    """Generates a Hilbert curve
    
    Parameters
    ----------
    n : int
        Order of the curve
        
    Returns
    -------
    points : ndarray
        Array of points generated
    """
    if n == 1:
        return np.array([
            [0, 0, 1, 1],
            [1, 0, 0, 1]
        ])
    else:
        points = hilbert_curve(n-1)
        points_top_left = _hilbert_top_left(points)
        points_bottom_left = _hilbert_bottom_left(points)
        points_bottom_right = _hilbert_bottom_right(points)
        points_top_right = _hilbert_top_right(points)
        return np.concatenate([
            points_top_left,
            points_bottom_left,
            points_bottom_right,
            points_top_right
        ], axis=1)

def sierpinski_triangle(n: int) -> np.ndarray:
    """Generates a Sierpinski triangle
    
    Parameters
    ----------
    n : int
        Order of the triangle
        
    Returns
    -------
    points : ndarray
        Array of points generated
    """
    if n == 1:
        return [np.array([
                [0, 1, 0.5, 0],
                [0, 0, 0.5*np.sqrt(3), 0]
            ])]
    else:
        triangles = sierpinski_triangle(n-1)
        triangles_top = [triangle.copy() for triangle in triangles]
        triangles_bottom_left = [triangle.copy() for triangle in triangles]
        triangles_bottom_right = [triangle.copy() for triangle in triangles]
        for triangle in triangles_top:
            triangle[0, :] += 2**(n-3)
            triangle[1, :] += 2**(n-3)*np.sqrt(3)
        for triangle in triangles_bottom_right:
            triangle[0, :] += 2**(n-2)
        return triangles_top + triangles_bottom_left + triangles_bottom_right

# rotate 90 degrees anti-clockwise with origin at last point
def _rotate90_dragon(points):
    # points is a 2xN array
    assert points.shape[0] == 2
    points_rot = np.array([-points[1]+points[1, -1], points[0]-points[0, -1]])
    points_rot[0, :] += points[0, -1]
    points_rot[1, :] += points[1, -1]
    return np.fliplr(points_rot)

def dragon_curve(n):
    """Generates a dragon curve
    
    Parameters
    ----------
    n : int
        Order of the curve
        
    Returns
    -------
    points : ndarray
        Array of points generated
    """
    if n == 0:
        return np.array([
            [0, 0],
            [0, 1]
       ], dtype=np.float64)
    else:
        points = dragon_curve(n-1)
        points_next = _rotate90_dragon(points)
        ratio = 1 / (2 + np.sqrt(2))
        points[:, -1] = points[:, -1] * (1-ratio) + points[:, -2] * ratio
        points_next[:, 0] = points_next[:, 0] * (1-ratio) + points_next[:, 1] * ratio
        return np.concatenate([points, points_next], axis=1)
