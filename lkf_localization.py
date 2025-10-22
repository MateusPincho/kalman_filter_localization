import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse # Para a elipse de incerteza

from robot import PointRobot
from LinearKalmanFilter import LinearKalmanFilter

# --- Parâmetros da Simulação ---
SIM_TIME = 10.0
DT = 0.05
NUM_STEPS = int(SIM_TIME / DT)

# --- Inicialização dos Objetos ---

# Parâmetros de controle (aceleração constante)
A_X = 0.1  # m/s^2
A_Y = 0   # m/s^2
u_control = np.array([A_X, A_Y])

# 4.1. Robô "Ground Truth"
# Começa em (0, 0) com velocidade (5, 2)
robot_gt = PointRobot(x0=0.0, y0=0.0, 
                      vx0=0.5, vy0=0.2, 
                      ax=A_X, ay=A_Y)

# Posição Inicial 
mu_inicial = np.array([0.1, 0.1, 0.5, 0.2])

# Incerteza inicial (P_0 ou Sigma_0)
P_inicial = np.diag([
    0.1**2,  # Incerteza std dev de 10cm em X
    0.1**2,  # Incerteza std dev de 10cm em Y
    0.05**2, # Incerteza std dev de 5cm/s em VX
    0.05**2  # Incerteza std dev de 5cm/s em VY
])

# Ruído do Processo (R_t ou Q) - 
# Isso modela o quanto nosso modelo de é ruim.
R_processo = np.diag([
    (0.01)**2, # Incerteza pos X (1cm)
    (0.01)**2, # Incerteza pos Y (1cm)
    (0.05)**2, # Incerteza vel X (5cm/s)
    (0.05)**2  # Incerteza vel Y (5cm/s)
]) * DT # Escalonamos pelo DT

estimator_kf = LinearKalmanFilter(mu0=mu_inicial, P0=P_inicial, R=R_processo)

# --- Execução da Simulação (Coleta de Dados) ---
for _ in range(NUM_STEPS):
    robot_gt.update(DT)
    estimator_kf.predict(u_control, DT)

# Pega os dados das trajetórias
gt_trajectory = robot_gt.get_trajectory_data()
est_trajectory, est_covariances = estimator_kf.get_belief_history()

# --- Configuração da Animação ---
fig, ax = plt.subplots()

# Define os limites do gráfico dinamicamente com uma margem
all_x = np.concatenate((gt_trajectory[0], est_trajectory[0]))
all_y = np.concatenate((gt_trajectory[1], est_trajectory[1]))
min_x, max_x = np.min(all_x), np.max(all_x)
min_y, max_y = np.min(all_y), np.max(all_y)
padding_x = (max_x - min_x) * 0.1
padding_y = (max_y - min_y) * 0.1

ax.set_xlim(min_x - padding_x, max_x + padding_x)
ax.set_ylim(min_y - padding_y, max_y + padding_y)

ax.set_xlabel("Posição X (m)")
ax.set_ylabel("Posição Y (m)")
ax.set_title("Predição do Filtro de Kalman (Sem Correção)")
ax.grid(True)
ax.set_aspect('equal')

# Objetos que serão animados:
robot_gt_plot, = ax.plot([], [], 'bo', markersize=10, label="Robô (Ground Truth)")
robot_est_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label="Robô (Estimado)")
traj_gt_plot, = ax.plot([], [], 'b--', label="Trajetória Real")
traj_est_plot, = ax.plot([], [], 'r--', label="Trajetória Estimada")

# A elipse de incerteza (começa vazia)
# 'CONFIDENCE_LEVEL' define o tamanho (ex: 2-sigma, 3-sigma)
# 2.4477 corresponde a ~95% de confiança para uma gaussiana 2D
CONFIDENCE_LEVEL = 2.4477 
unc_ellipse = Ellipse(xy=(0,0), width=0, height=0, angle=0, 
                      facecolor='red', alpha=0.2, label="Incerteza (95%)")
ax.add_patch(unc_ellipse)
ax.legend()


def get_ellipse_params(P_matrix_4x4):
    """ Extrai parâmetros da elipse da matriz de covariância 4x4. """
    # Pegamos apenas a sub-matriz 2x2 referente à posição (px, py)
    P_pos = P_matrix_4x4[0:2, 0:2]
    
    # Calcula autovetores e autovalores
    # Autovalores (eigvals) dão a variância nas direções principais
    # Autovetores (eigvecs) dão as direções (orientação da elipse)
    eigvals, eigvecs = np.linalg.eigh(P_pos)
    
    # Os eixos da elipse são baseados no desvio padrão (sqrt(variância))
    std_dev_1 = np.sqrt(eigvals[0])
    std_dev_2 = np.sqrt(eigvals[1])
    
    # Largura e Altura da elipse (escaladas pelo nível de confiança)
    width = 2 * CONFIDENCE_LEVEL * std_dev_1
    height = 2 * CONFIDENCE_LEVEL * std_dev_2
    
    # Ângulo da elipse
    # O primeiro autovetor (coluna 0) nos dá a direção do primeiro eixo
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    
    return width, height, angle

# Função 'animate' (chamada para cada frame)
def animate(i):
    # Atualiza robô ground truth
    robot_gt_plot.set_data([gt_trajectory[0, i]], [gt_trajectory[1, i]])
    traj_gt_plot.set_data(gt_trajectory[0, :i+1], gt_trajectory[1, :i+1])
    
    # Atualiza robô estimado
    robot_est_plot.set_data([est_trajectory[0, i]], [est_trajectory[1, i]])
    traj_est_plot.set_data(est_trajectory[0, :i+1], est_trajectory[1, :i+1])

    # Atualiza a elipse de incerteza
    pos_estimada = est_trajectory[:, i]
    cov_estimada = est_covariances[i]
    
    width, height, angle = get_ellipse_params(cov_estimada)
    
    unc_ellipse.set_center(pos_estimada)
    unc_ellipse.set_width(width)
    unc_ellipse.set_height(height)
    unc_ellipse.set_angle(angle) # set_angle espera graus
    
    return (robot_gt_plot, traj_gt_plot, 
            robot_est_plot, traj_est_plot, unc_ellipse)

# --- 7. Rodar a Animação ---
ani = animation.FuncAnimation(fig, animate, frames=NUM_STEPS + 1,
                              init_func=lambda: animate(0), # init limpa a tela
                              blit=True,
                              interval=DT * 1000, repeat=False)

plt.show()