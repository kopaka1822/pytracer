import numpy as np

class Plane:

    def __init__(self, P1: np.ndarray, P2: np.ndarray, ior: float = 1.0):
        self._P1 = np.asarray(P1, dtype=float)
        self._P2 = np.asarray(P2, dtype=float)
        # Richtungsvektor vom Segment
        dir_vec = self._P2 - self._P1
        # 90° CCW Rotation: (x, y) -> (-y, x)
        normal = np.array([-dir_vec[1], dir_vec[0]], dtype=float)
        normal /= np.linalg.norm(normal)
        self._normal = normal
        self._ior = float(ior)

    # --- Getter ---
    def P(self) -> np.ndarray:
        return self._P1

    def P1(self) -> np.ndarray:
        return self._P1

    def P2(self) -> np.ndarray:
        return self._P2

    def N(self) -> np.ndarray:
        return self._normal

    def Ior(self) -> float:
        return self._ior

    def __repr__(self) -> str:
        return f"Plane(P1={self._P1}, P2={self._P2}, N={self._normal}, IOR={self._ior})"
