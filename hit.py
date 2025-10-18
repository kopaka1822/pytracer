import numpy as np
from plane import Plane

class Hit:

    def __init__(self, plane: Plane, point: np.ndarray, t: float):
        self._plane = plane
        self._point = np.asarray(point, dtype=float)
        self._t = float(t)

    def Plane(self) -> Plane:
        return self._plane

    def P(self) -> np.ndarray:
        return self._point

    def T(self) -> float:
        return self._t

    def __repr__(self) -> str:
        return f"Hit(P={self._point}, t={self._t}, Plane={self._plane})"
