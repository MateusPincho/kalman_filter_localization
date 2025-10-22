import numpy as np

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
        
        # 2. Equação de Predição do Estado 
        # mu_bar = A_t * mu_{t-1} + B_t * u_t
        
        self.mu = (self.F @ self.mu) + (self.G @ u)
        
        # 3. Equação de Predição da Covariância (Equação 2 da imagem)
        # Sigma_bar = A_t * Sigma_{t-1} * A_t' + R_t
        self.P = (self.F @ self.P @ self.F.T) + self.R
        
        # 4. Correção do Estado 

        
        # Salva no histórico
        self.mu_history.append(self.mu.copy())
        self.P_history.append(self.P.copy())

    def get_belief_history(self):
        # Retorna posições [x, y] e covariâncias
        positions = np.array(self.mu_history).reshape(-1, 4)[:, :2].T
        covariances = self.P_history
        return positions, covariances
