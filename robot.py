import numpy as np

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