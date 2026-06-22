import numpy as np
from scipy.stats import gaussian_kde

def draw_density_blob(ax, points, quantile=0.25, bw=0.3, pad=0.6,
                      gridsize=250, color='black', linewidth=2.0):
    """
    Desenha um contorno suave em volta de uma região densa usando KDE.

    points: array (n, 2)
    quantile: controla o tamanho da região
        - menor -> contorno maior
        - maior -> contorno mais colado no núcleo denso
    bw: bandwidth da KDE
        - menor -> contorno mais irregular
        - maior -> contorno mais suave
    """
    kde = gaussian_kde(points.T, bw_method=bw)

    x_min, y_min = points.min(axis=0) - pad
    x_max, y_max = points.max(axis=0) + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, gridsize),
        np.linspace(y_min, y_max, gridsize)
    )

    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)

    # nível baseado na densidade observada nos próprios pontos
    point_density = kde(points.T)
    level = np.quantile(point_density, quantile)

    cs = ax.contour(
        xx, yy, zz,
        levels=[level],
        colors=color,
        linewidths=linewidth
    )

    return cs


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans

# ============================================================
# Configuração geral
# ============================================================
rng = np.random.default_rng(7)


# ============================================================
# Funções auxiliares
# ============================================================
def sample_gaussian_layers(center, n_core=90, n_shell=30,
                           core_std=0.30, shell_std=0.75, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    core = rng.normal(loc=center, scale=core_std, size=(n_core, 2))
    shell = rng.normal(loc=center, scale=shell_std, size=(n_shell, 2))
    return core, shell


def sample_annulus(center, n, r_min, r_max, angle_range=(0, 2*np.pi), rng=None):
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.uniform(angle_range[0], angle_range[1], n)
    r = np.sqrt(rng.uniform(r_min**2, r_max**2, n))

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack([x, y])


def sample_irregular_blob(
    center,
    n=60,
    base_radius=1.0,
    noise=0.05,
    stretch_x=1.15,
    stretch_y=0.85,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.uniform(0, 2*np.pi, n)

    radial_shape = (
        base_radius
        + 0.22 * np.sin(3 * theta + 0.5)
        + 0.16 * np.sin(5 * theta - 0.8)
        + rng.normal(0, noise, n)
    )
    radial_shape = np.clip(radial_shape, 0.35, None)

    u = np.sqrt(rng.uniform(0, 1, n))
    r = radial_shape * u

    x = center[0] + stretch_x * r * np.cos(theta)
    y = center[1] + stretch_y * r * np.sin(theta)

    return np.column_stack([x, y])


def min_distance(point, cloud):
    return np.min(np.linalg.norm(cloud - point, axis=1))


def prevent_circle_overlap(circle_infos, gap=0.12):
    if len(circle_infos) != 2:
        return circle_infos

    (center_left, radius_left), (center_right, radius_right) = circle_infos
    center_distance = np.linalg.norm(center_right - center_left)
    allowed_sum = max(center_distance - gap, 0.0)
    current_sum = radius_left + radius_right

    if current_sum <= allowed_sum or current_sum == 0:
        return circle_infos

    shrink = allowed_sum / current_sum
    return [
        (center_left, radius_left * shrink),
        (center_right, radius_right * shrink),
    ]


def points_inside_any_circle(points, circle_infos):
    inside_mask = np.zeros(len(points), dtype=bool)

    for center, radius in circle_infos:
        distances = np.linalg.norm(points - center, axis=1)
        inside_mask |= distances <= radius

    return inside_mask


# ============================================================
# 1) Construção dos dados
# ============================================================
center1 = np.array([0.0, 0.0])

core1, shell1 = sample_gaussian_layers(
    center=center1,
    n_core=90,
    n_shell=28,
    core_std=0.53,
    shell_std=0.75,
    rng=rng
)

omega1_density = np.vstack([core1, shell1])

noise1 = np.vstack([
    sample_annulus(
        center=center1,
        n=12,
        r_min=1.8,
        r_max=3.6,
        angle_range=(0.70*np.pi, 1.55*np.pi),
        rng=rng
    ),
    sample_annulus(
        center=center1,
        n=7,
        r_min=2.1,
        r_max=3.2,
        angle_range=(1.60*np.pi, 1.95*np.pi),
        rng=rng
    )
])

center2 = np.array([5.0, -0.5])

# omega2 original continua sendo usado no KMeans
omega2 = sample_irregular_blob(
    center=center2,
    n=90,
    base_radius=0.92,
    noise=0.08,
    stretch_x=1.85,
    stretch_y=0.45,
    rng=rng
)

# pontos extras só para alargar visualmente a mancha da direita
omega2_right_tail = sample_irregular_blob(
    center=center2 + np.array([-0.38, 0.00]),  # controla o quanto vai para a direita
    n=24,                                     # controla o quanto essa "cauda" pesa
    base_radius=0.36,
    noise=0.015,
    stretch_x=1.65,
    stretch_y=0.32,
    rng=rng
)

# conjunto só para desenhar a região de densidade
omega2_density_visual = np.vstack([omega2, omega2_right_tail])

# ruído interno extra da região da direita:
# entra no KMeans, mas não no contorno azul de densidade
noise2_inner = center2 + sample_annulus(
    center=np.array([0.0, 0.0]),
    n=18,
    r_min=0.75,
    r_max=1.55,
    rng=rng
) * np.array([1.15, 0.42])



# ============================================================
# ponto z_i com controle manual
# ============================================================
zi_dx = 2.6
zi_dy = -0.15

zi = center1 + np.array([zi_dx, zi_dy])

# Todos os pontos usados no KMeans
all_points = np.vstack([core1, shell1, noise1, omega2, noise2_inner, zi.reshape(1, 2)])


# ============================================================
# 2) KMeans com 2 grupos
# ============================================================
kmeans = KMeans(n_clusters=2, random_state=7, n_init=20)
labels = kmeans.fit_predict(all_points)
centers = kmeans.cluster_centers_

# ordenar clusters da esquerda para a direita
order = np.argsort(centers[:, 0])


# ============================================================
# 3) Plot
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

point_style = dict(
    s=10,
    facecolors='none',
    edgecolors='black',
    linewidths=1.0,
    alpha=0.3
)

# destacar z_i
ax.scatter(*zi, c='red', s=20, marker='o', zorder=5)
ax.text(zi[0] - 0.10, zi[1] + 0.08, r'$z_i$', fontsize=13, c='red', ha='right')

# guardar info dos círculos
circle_infos = []

# calcular raio de cada cluster
for new_idx, old_k in enumerate(order, start=1):
    cluster_points = all_points[labels == old_k]
    center = centers[old_k]

    distances = np.linalg.norm(cluster_points - center, axis=1)
    radius = distances.max()

    circle_infos.append((center, radius))

# evita interseccao quando os centroides ficam mais proximos
circle_infos = prevent_circle_overlap(circle_infos)

# plota apenas os pontos que continuam dentro de pelo menos um dos circulos
visible_points = all_points[points_inside_any_circle(all_points, circle_infos)]
ax.scatter(visible_points[:, 0], visible_points[:, 1], **point_style)

# desenhar círculo de cada cluster
for new_idx, (center, radius) in enumerate(circle_infos, start=1):
    circle = Circle(
        xy=center,
        radius=radius,
        fill=False,
        edgecolor='green',
        linewidth=1.5,
        linestyle='--',
        alpha=0.5
    )
    ax.add_patch(circle)

    # centróide
    ax.scatter(center[0], center[1], c='red', s=20, marker='o', zorder=6)

mu1_center = circle_infos[0][0]
ax.plot(
    [zi[0], mu1_center[0]],
    [zi[1], mu1_center[1]],
    color='red',
    linestyle='--',
    linewidth=1.4,
    alpha=0.8,
    zorder=4
)


# contorno da região densa da esquerda
draw_density_blob(
    ax,
    core1,
    quantile=0.1,
    bw=0.5,
    color='blue',
    linewidth=2.0
)

# contorno da região densa da direita
draw_density_blob(
    ax,
    omega2_density_visual,
    quantile=0.04,
    bw=0.58,
    color='blue',
    linewidth=2.0
)

# ============================================================
# 4) Rótulos "fora" das regiões
# ============================================================

# --- KMEANS: um rótulo perto de cada círculo ---
(left_center, left_radius), (right_center, right_radius) = circle_infos

ax.text(
    left_center[0] - 0.15,
    left_center[1] + left_radius + 0.45,
    r'$\Omega_{\text{KMeans}_1}$',
    color='green',
    fontsize=13,
    fontweight='bold',
    ha='center',
    va='bottom'
)

ax.text(
    right_center[0],
    right_center[1] + right_radius + 0.35,
    r'$\Omega_{\text{KMeans}_2}$',
    color='green',
    fontsize=13,
    fontweight='bold',
    ha='center',
    va='bottom'
)

# --- DENSITY: um rótulo perto de cada mancha ---
core1_center = core1.mean(axis=0)
omega2_center = omega2.mean(axis=0)

ax.text(
    core1_center[0],
    core1[:, 1].max() - 2.85,
    r'$\Omega_{\text{Density}_1}$',
    color='blue',
    fontsize=13,
    fontweight='bold',
    ha='center',
    va='bottom'
)

ax.text(
    omega2_center[0],
    omega2[:, 1].min() - 0.45,
    r'$\Omega_{\text{Density}_2}$',
    color='blue',
    fontsize=13,
    fontweight='bold',
    ha='center',
    va='top'
)

ax.set_aspect('equal')
ax.set_xlim(-4.1, 7.9)
ax.set_ylim(-4.5, 4.2)
ax.axis('off')

plt.tight_layout()
plt.savefig("fig_artigo.png")
plt.show()
