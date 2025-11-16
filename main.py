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

def get_point_coordinate(point):
    x_pos = point % (n+1)
    y_pos = point // (n+1)
    return np.array([x_pos/n, y_pos/n])

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
    A = np.zeros((dof, dof), dtype=float)
    for i, m in enumerate(mu):
        points = get_points_of_triangle(i)
        if i % 2:
            A[points[0], points[0]] += 1*m
            A[points[0], points[2]] += -1*m
            A[points[2], points[0]] += -1*m
            A[points[1], points[2]] += -1*m
            A[points[2], points[1]] += -1*m
            A[points[2], points[2]] += 2*m
        else:
            A[points[0], points[0]] += 2*m
            A[points[0], points[1]] += -1*m
            A[points[0], points[2]] += -1*m
            A[points[1], points[0]] += -1*m
            A[points[2], points[0]] += -1*m
            A[points[2], points[2]] += 1*m
        A[points[1], points[1]] += 1*m
    A *= 1/2
    A += sigma * np.eye(dof)

    return A


def make_B_matrix(u):
    dof = 2*n*n
    B = np.zeros((dof, dof), dtype=float)

    for i in range(dof):
        points = get_points_of_triangle(i)
        vec = np.array([0, 0], dtype=float)
        if i % 2:
            vec += u[points[0]] * np.array([0, -1], dtype=float)
            vec += u[points[1]] * np.array([-1, 0], dtype=float)
            vec += u[points[2]] * np.array([1, 1], dtype=float)
        else:
            vec += u[points[0]]*np.array([-1, -1], dtype=float)
            vec += u[points[1]]*np.array([1, 0], dtype=float)
            vec += u[points[2]]*np.array([0, 1], dtype=float)
        B[i, i] = np.linalg.norm(vec) - 1

    B *= 1/2*dx
    return B

def make_b_vector(f):
    dof = (n+1)*(n+1)
    b = np.zeros(dof, dtype=float)

    for i in range(dof):
        coords = get_point_coordinate(i)
        b[i] = f(coords[0], coords[1])

    b *= 1/2*dx*dx
    return b



def solve_system(f, mu_0, dt, T):
    mu = mu_0.copy()
    u = np.zeros((n+1)*(n+1), dtype=float)

    t = 0

    while t < T:
        A = make_A_matrix(mu)
        b = make_b_vector(f)
        u = np.linalg.solve(A, b)
        B = make_B_matrix(u)
        mu = mu + dt * B @ mu
        t += dt
    return u, mu

def f(x, y):
    mask1 = np.bitwise_and(abs(x-0.3) < 0.1, abs(y-0.5) < 0.2)
    mask2 = np.bitwise_and(abs(x-0.7) < 0.1, abs(y-0.5) < 0.2)
    res = np.zeros_like(x)
    res[mask1] = 1.0
    res[mask2] = -1.0
    return res

def mu_0():
    dof = 2*n*n
    mu = np.ones(dof, dtype=float)
    return mu

import matplotlib.tri as mtri

# assumes: n, get_points_of_triangle, get_point_coordinate are defined globally

def build_mesh():
    """Return node coordinates (x,y) and triangle connectivity."""
    # nodes
    num_points = (n + 1) * (n + 1)
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    for p in range(num_points):
        coord = get_point_coordinate(p)
        x[p] = coord[0]
        y[p] = coord[1]

    # triangles
    num_tris = 2 * n * n
    triangles = np.zeros((num_tris, 3), dtype=int)
    for i in range(num_tris):
        triangles[i, :] = get_points_of_triangle(i)

    return x, y, triangles


def compute_mu_grad_u(u, mu, x, y, triangles):
    """
    Compute piecewise-constant mu * grad(u) per triangle.

    Returns:
        tri_centers_x, tri_centers_y, vx, vy
    """
    num_tris = triangles.shape[0]
    vx = np.zeros(num_tris)
    vy = np.zeros(num_tris)
    cx = np.zeros(num_tris)
    cy = np.zeros(num_tris)

    for i in range(num_tris):
        p0, p1, p2 = triangles[i]
        x0, y0 = x[p0], y[p0]
        x1, y1 = x[p1], y[p1]
        x2, y2 = x[p2], y[p2]
        u0, u1, u2 = u[p0], u[p1], u[p2]

        # triangle area (2*area)
        detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area2 = detJ
        if area2 == 0:
            continue

        # gradients of nodal basis functions in physical coords
        # See standard P1 element formulas:
        # grad(phi0) = [ (y1 - y2), (x2 - x1) ] / (2*area)
        # grad(phi1) = [ (y2 - y0), (x0 - x2) ] / (2*area)
        # grad(phi2) = [ (y0 - y1), (x1 - x0) ] / (2*area)
        area = area2 / 2.0

        grad_phi0 = np.array([(y1 - y2), (x2 - x1)]) / (2.0 * area)
        grad_phi1 = np.array([(y2 - y0), (x0 - x2)]) / (2.0 * area)
        grad_phi2 = np.array([(y0 - y1), (x1 - x0)]) / (2.0 * area)

        grad_u = u0 * grad_phi0 + u1 * grad_phi1 + u2 * grad_phi2

        # multiply by mu on this triangle
        vec = mu[i] * grad_u
        vx[i], vy[i] = vec[0], vec[1]

        # centroid for plotting the vector
        cx[i] = (x0 + x1 + x2) / 3.0
        cy[i] = (y0 + y1 + y2) / 3.0

    return cx, cy, vx, vy


def plot_solution(u, mu):
    """
    Create 3 plots:
    1. mu (piecewise constant on triangles)
    2. u (piecewise linear on nodes)
    3. vector field mu * grad(u) (piecewise constant per triangle)
    """
    x, y, triangles = build_mesh()
    triang = mtri.Triangulation(x, y, triangles)

    # prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) mu
    ax = axes[0]
    tpc = ax.tripcolor(triang, facecolors=mu, edgecolors='k')
    ax.set_title(r'$\mu$')
    ax.set_aspect('equal')
    fig.colorbar(tpc, ax=ax)

    # 2) u
    ax = axes[1]
    tpc2 = ax.tripcolor(triang, u, shading='gouraud', edgecolors='k')
    ax.set_title(r'$u$')
    ax.set_aspect('equal')
    fig.colorbar(tpc2, ax=ax)

    # 3) mu * grad(u)
    cx, cy, vx, vy = compute_mu_grad_u(u, mu, x, y, triangles)
    ax = axes[2]
    ax.tripcolor(triang, facecolors=mu, alpha=0.2, edgecolors='k')
    ax.quiver(cx, cy, vx, vy, angles='xy', scale_units='xy', scale=1.0)
    ax.set_title(r'$\mu \nabla u$')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    n = 20
    dx = 1.0 / n
    dt = 0.01
    T = 10
    sigma = 1e-10
    mu_initial = mu_0()
    u, mu = solve_system(f, mu_initial, dt, T)
    plot_solution(u, mu)
