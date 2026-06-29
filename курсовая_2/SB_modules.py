import numpy as np
from scipy.sparse import csr_matrix


class SBBase:
    """
    Базовый класс для всех вариантов Simulated Bifurcation (BSB, DSB).

    Здесь:
    - храним матрицу J
    - инициализируем x и y
    - задаём общий цикл по итерациям
    """

    def __init__(self, J, h=None, n_iter=1000, dt=1.0, xi=None, seed=None):
        """
        Параметры:
        J      : разреженная (или плотная) матрица связей (N x N), обычно J = -W
        h      : внешнее поле (если None, то поле отсутствует)
        n_iter : число итераций
        dt     : шаг интегрирования
        xi     : коэффициент перед матрицей J (если None, ставим стандартное значение из QAIA)
        seed   : сид для генератора случайных чисел
        """
        if seed is not None:
            np.random.seed(seed)

        self.J = csr_matrix(J, dtype=float)
        self.N = self.J.shape[0]

        if h is None:
            self.h = np.zeros((self.N, 1), dtype=float)
        else:
            h = np.asarray(h, dtype=float).reshape(-1, 1)
            if h.shape[0] != self.N:
                raise ValueError(f"h has wrong shape: {h.shape}, expected ({self.N}, 1)")
            self.h = h

        self.n_iter = n_iter
        self.dt = float(dt)

        # параметр "детюнинг" delta и линейный рост накачки p(t)
        self.delta = 1.0
        self.p = np.linspace(0.0, 1.0, self.n_iter)

        # коэффициент xi (масштаб взаимодействия)
        if xi is None:
            # ровно та же формула, что в оригинальном коде
            den = float(self.J.multiply(self.J).sum())
            den = np.sqrt(den) if den > 0 else 1.0
            self.xi = 0.5 * np.sqrt(self.N - 1) / den
        else:
            self.xi = float(xi)

        # начальные условия для x и y (маленькие случайные значения)
        self.x = 0.02 * (np.random.rand(self.N, 1) - 0.5)
        self.y = 0.02 * (np.random.rand(self.N, 1) - 0.5)

    def run(self, record_trajectory=False):
        """
        Основной цикл.
        Если record_trajectory=True, возвращаем массив траекторий x[t, i].
        """
        # if record_trajectory:
        #     traj = np.zeros((self.n_iter + 1, self.N))
        #     traj[0, :] = self.x[:, 0]

        for t in range(self.n_iter):
            self.update_step(t)
            if record_trajectory:
                traj[t + 1, :] = self.x[:, 0]

        # if record_trajectory:
        #     return traj

    def update_step(self, t):
        """
        Один шаг по времени.
        В базовом классе не реализован, каждая конкретная схема (BSB/DSB)
        реализует его по-своему.
        """
        raise NotImplementedError("Нужно переопределить update_step в подклассе.")


# -----------------------------
# 5. Ballistic SB (BSB)
# -----------------------------
class BSB(SBBase):
    """
    Ballistic Simulated Bifurcation.

    Отличие от ASB:
    - Нет кубической нелинейности.
    - Вводится «жёсткое насыщение» |x_i| <= 1: при превышении клипуем x_i и обнуляем y_i.
    """

    def update_step(self, t):
        """
        Один шаг по времени для BSB (модифицированный симплектический Эйлер).

        y_{t+1} = y_t + dt * [ - (delta - p_t) * x_t + xi * J x_t ]
        x_{t+1} = x_t + dt * delta * y_{t+1}

        затем:
        если |x_i| > 1, то
            x_i = sign(x_i)
            y_i = 0
        """
        p_t = self.p[t]

        # Сначала обновляем импульс y
        self.y = self.y + (-(self.delta - p_t) * self.x + self.xi * (self.J.dot(self.x) + self.h)) * self.dt

        # Потом обновляем координату x
        self.x = self.x + self.dt * self.y * self.delta

        # Насыщение: ограничиваем |x| <= 1
        cond = np.abs(self.x) > 1.0
        self.x = np.where(cond, np.sign(self.x), self.x)
        self.y = np.where(cond, 0.0, self.y)


# -----------------------------
# 6. Discrete SB (DSB)
# -----------------------------
class DSB(SBBase):
    """
    Discrete Simulated Bifurcation.

    Отличие от BSB:
    - В J.dot() подаём уже дискретизированные спины sign(x).
    """

    def update_step(self, t):
        """
        Один шаг по времени для DSB.

        Здесь в силу знака x мы постепенно приближаемся к ±1.
        y_{t+1} = y_t + dt * [ - (delta - p_t) * x_t + xi * J sign(x_t) ]
        x_{t+1} = x_t + dt * delta * y_{t+1}

        далее, как и в BSB, делаем насыщение:
            если |x_i| > 1, то x_i = sign(x_i), y_i = 0.
        """
        p_t = self.p[t]
        sign_x = np.sign(self.x)

        self.y = self.y + (-(self.delta - p_t) * self.x + self.xi * (self.J.dot(sign_x) + self.h)) * self.dt
        self.x = self.x + self.dt * self.y * self.delta

        cond = np.abs(self.x) > 1.0
        self.x = np.where(cond, np.sign(self.x), self.x)
        self.y = np.where(cond, 0.0, self.y)
