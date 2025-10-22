import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse # Para a elipse de incerteza

# --- 1. CLASSE DO ROBÔ (GROUND TRUTH) ---
# Esta classe representa o robô real, perfeito, no mundo.
# (Exatamente como a anterior)
class PointRobot:
    def __init__(self, x0, y0, vx0, vy0, ax, ay):
        self.state = np.array([x0, y0, vx0, vy0], dtype=float)
        self.acceleration = np.array([ax, ay], dtype=float)
        self.history = [self.state[:2].copy()] # Histórico apenas da posição [x, y]

    def update(self, dt):
        # Pega o estado atual
        px, py, vx, vy = self.state
        ax, ay = self.acceleration

        # p_k+1 = p_k + v_k*dt + 0.5*a*(dt^2)
        px_new = px + vx * dt + 0.5 * ax * (dt**2)
        py_new = py + vy * dt + 0.5 * ay * (dt**2)
        
        # v_k+1 = v_k + a*dt
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        
        # Atualiza o vetor de estado
        self.state = np.array([px_new, py_new, vx_new, vy_new])
        self.history.append(self.state[:2].copy())
    
    def get_trajectory_data(self):
        return np.array(self.history).T

# --- 2. CLASSE DO FILTRO DE KALMAN ---
# Esta classe representa a CRENÇA (belief) do robô sobre seu estado.
class LinearKalmanFilter:
    """
    Implementa um Filtro de Kalman Linear focado apenas no passo de PREDIÇÃO.
    O estado é x = [px, py, vx, vy]'
    """
    def __init__(self, mu0, P0, R):
        """
        Args:
            mu0 (np.array): Estado inicial estimado [4x1] (nosso mu_0)
            P0 (np.array): Covariância inicial [4x4] (nossa Sigma_0)
            R (np.array): Covariância do Ruído do Processo [4x4] (nossa R_t)
                          (Representa a incerteza do modelo)
        """
        self.mu = mu0.reshape(4, 1)  # Vetor de estado [px, py, vx, vy]
        self.P = P0                  # Matriz de covariância (Sigma)
        self.R = R                   # Ruído do processo
        
        # Matrizes F (A_t) e G (B_t) dependem de dt, serão criadas no predict
        self.F = np.eye(4)
        self.G = np.zeros((4, 2))
        
        # Histórico da crença
        self.mu_history = [self.mu.copy()]
        self.P_history = [self.P.copy()]

    def _build_matrices(self, dt):
        """ Constrói as matrizes F e G com base em dt """
        # F (ou A_t) - Matriz de Transição de Estado
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # G (ou B_t) - Matriz de Entrada de Controle
        self.G = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])

    def predict(self, u, dt):
        """
        Executa o passo de predição do Filtro de Kalman.
        
        Args:
            u (np.array): Vetor de controle [2x1] (nossa u_t = [ax, ay])
            dt (float): Passo de tempo
        """
        # Garante que u é um vetor coluna [2x1]
        u = u.reshape(2, 1)
        
        # 1. Atualiza as matrizes F e G para o dt atual
        self._build_matrices(dt)
        
        # 2. Equação de Predição do Estado (Equação 1 da imagem)
        # mu_bar = A_t * mu_{t-1} + B_t * u_t
        # (Usando F e G da segunda imagem)
        self.mu = (self.F @ self.mu) + (self.G @ u)
        
        # 3. Equação de Predição da Covariância (Equação 2 da imagem)
        # Sigma_bar = A_t * Sigma_{t-1} * A_t' + R_t
        self.P = (self.F @ self.P @ self.F.T) + self.R
        
        # Salva no histórico
        self.mu_history.append(self.mu.copy())
        self.P_history.append(self.P.copy())

    def get_belief_history(self):
        # Retorna posições [x, y] e covariâncias
        positions = np.array(self.mu_history).reshape(-1, 4)[:, :2].T
        covariances = self.P_history
        return positions, covariances

# --- 3. Parâmetros da Simulação ---
SIM_TIME = 10.0
DT = 0.05
NUM_STEPS = int(SIM_TIME / DT)

# --- 4. Inicialização dos Objetos ---

# Parâmetros de controle (aceleração constante)
A_X = 0  # m/s^2
A_Y = 0   # m/s^2
u_control = np.array([A_X, A_Y])

# 4.1. Robô "Ground Truth"
# Começa em (0, 0) com velocidade (5, 2)
robot_gt = PointRobot(x0=0.0, y0=0.0, 
                      vx0=0.5, vy0=0.2, 
                      ax=A_X, ay=A_Y)

# 4.2. Filtro de Kalman (Nossa Crença)
# Vamos supor que o robô *acha* que começa em (0.1, 0.1)
# (Um pequeno erro inicial)
mu_inicial = np.array([0.1, 0.1, 0.5, 0.2])

# Incerteza inicial (P_0 ou Sigma_0)
# Uma pequena incerteza inicial na posição e velocidade
P_inicial = np.diag([
    0.1**2,  # Incerteza std dev de 10cm em X
    0.1**2,  # Incerteza std dev de 10cm em Y
    0.05**2, # Incerteza std dev de 5cm/s em VX
    0.05**2  # Incerteza std dev de 5cm/s em VY
])

# Ruído do Processo (R_t ou Q) - O MAIS IMPORTANTE
# Isso modela o quanto nosso modelo de "aceleração constante" é ruim.
# Vamos supor que a aceleração pode variar um pouco (ruído) a cada passo.
# Adicionamos ruído na posição e velocidade
R_processo = np.diag([
    (0.01)**2, # Incerteza pos X adicionada a cada passo (1cm)
    (0.01)**2, # Incerteza pos Y adicionada a cada passo (1cm)
    (0.05)**2, # Incerteza vel X adicionada a cada passo (5cm/s)
    (0.05)**2  # Incerteza vel Y adicionada a cada passo (5cm/s)
]) * DT # Escalonamos pelo DT

estimator_kf = LinearKalmanFilter(mu0=mu_inicial, P0=P_inicial, R=R_processo)

# --- 5. Execução da Simulação (Coleta de Dados) ---
for _ in range(NUM_STEPS):
    robot_gt.update(DT)
    estimator_kf.predict(u_control, DT)

# Pega os dados das trajetórias
gt_trajectory = robot_gt.get_trajectory_data()
est_trajectory, est_covariances = estimator_kf.get_belief_history()

# --- 6. Configuração da Animação ---
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