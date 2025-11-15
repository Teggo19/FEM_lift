import numpy as np
import matplotlib.pyplot as plt

# make a grid of triangles
def get_points_of_triangle(triangle):
    points = []
    x_pos = triangle % (2*n)
    y_pos = triangle // (2*n)
    if triangle % 2 == 0:
        points.append(y_pos*(n+1) + (x_pos)//2)
        points.append(y_pos*(n+1) + (x_pos)//2 + 1)
        points.append((y_pos+1)*(n+1) + (x_pos)//2)
    else:
        points.append(y_pos*(n+1) + (x_pos-1)//2 + 1)
        points.append((y_pos + 1)*(n+1) + (x_pos-1)//2)
        points.append((y_pos + 1)*(n+1) + (x_pos-1)//2 + 1)

    return points

def get_triangle_points(point):
    triangles = []
    x_pos = point % (n+1)
    y_pos = point // (n+1)
    if y_pos > 0:
        if x_pos > 0:
            triangles.append((y_pos-1)*(2*n) + x_pos*2-1) # Triangle to the lower left
        if x_pos < n:
            triangles.append((y_pos-1)*(2*n) + x_pos*2) # Triangles to the lower right
            triangles.append((y_pos-1)*(2*n) + x_pos*2+1)
    if y_pos < n:
        if x_pos > 0:
            triangles.append((y_pos)*(2*n) + x_pos*2-2)
            triangles.append((y_pos)*(2*n) + x_pos*2-1)
        if x_pos < n:
            triangles.append((y_pos)*(2*n) + x_pos*2)

    return triangles


def make_A_matrix(mu):
    dof = (n+1)*(n+1)
    A = np.zeros((dof, dof))




if __name__=='__main__':
    n = 5
    triangles = [5, 15, 25, 17]
    points = [17, 35, 32, 0]

    for triangle in triangles:
        print(f"Triangle {triangle} has points: {get_points_of_triangle(triangle)}")
    for point in points:
        print(f"Point {point} is part of triangles: {get_triangle_points(point)}")
