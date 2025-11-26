import numpy as np
from plane import Plane

class Hit:

    def __init__(self, plane: Plane, point: np.ndarray, t: float):
        self._plane = plane
        self._point = np.asarray(point, dtype=float)
        self._t = float(t)
        self._forcedNormal = None # forced shading normal for virtual plane intersections

    def Plane(self) -> Plane:
        return self._plane
    
    def ShadingN(self) -> np.ndarray:
        if self._forcedNormal is not None:
            return self._forcedNormal
        return self._plane.ShadingN(self._point)
    
    def CalcDN(self, dP: np.ndarray) -> np.ndarray:
        if self._forcedNormal is not None:
            return 0.0
        return self._plane.CalcDN(self._point, dP)
    
    def Tangent(self) -> np.ndarray:
        return self._plane.Tangent()

    def P(self) -> np.ndarray:
        return self._point

    def T(self) -> float:
        return self._t
    
    def overwriteShadingN(self, N: np.ndarray):
        self._forcedNormal = N / np.linalg.norm(N)

    def __repr__(self) -> str:
        return f"Hit(P={self._point}, t={self._t}, Plane={self._plane})"
