import numpy as np

class Plane:
    # ior: 1.0 = only reflection, 0.0 = absorb, 0<ior<1.0 and ior>1.0 refraction
    def __init__(self, P1: np.ndarray, P2: np.ndarray, ior: float = 1.0, P0: np.ndarray = None, P3: np.ndarray = None):
        self._P1 = np.asarray(P1, dtype=float)
        self._P2 = np.asarray(P2, dtype=float)

        if P0 is None: P0 = self._P1
        if P3 is None: P3 = self._P2

        dir = self._P2 - self._P1
        dir1 = self._P2 - P0
        dir2 = P3 - self._P1

        # 90° CCW Rotation: (x, y) -> (-y, x)
        normal = np.array([-dir[1], dir[0]], dtype=float)
        normal1 = np.array([-dir1[1], dir1[0]], dtype=float)
        normal2 = np.array([-dir2[1], dir2[0]], dtype=float)

        self._normal = normal / np.linalg.norm(normal) # actual surface normal
        self._ior = float(ior)
        self._normal1 = normal1 / np.linalg.norm(normal1) # shading normal 1
        self._normal2 = normal2 / np.linalg.norm(normal2) # shading normal 2

    # --- Getter ---
    def P(self) -> np.ndarray:
        return self._P1

    def P1(self) -> np.ndarray:
        return self._P1

    def P2(self) -> np.ndarray:
        return self._P2

    def N(self) -> np.ndarray:
        return self._normal
    
    def Tangent(self) -> np.ndarray:
        dir_vec = self._P2 - self._P1
        return dir_vec / np.linalg.norm(dir_vec)
    
    # unnormalized shading normal
    def UnnormalizedShadingN(self, P: np.ndarray) -> np.ndarray:
        # Linear interpolation/extrapolation of shading normal based on position P
        dir_vec = self._P2 - self._P1
        total_length_sq = np.dot(dir_vec, dir_vec)
        
        if total_length_sq == 0:
            return self._normal1  # Avoid division by zero

        # Use dot product to get signed distance along the plane direction
        vec_to_P = P - self._P1
        t = np.dot(vec_to_P, dir_vec) / total_length_sq

        shading_normal = (1 - t) * self._normal1 + t * self._normal2
        return shading_normal

    # normalized shading normal
    def ShadingN(self, P: np.ndarray) -> np.ndarray:
        unnormalized_N = self.UnnormalizedShadingN(P)
        return unnormalized_N / np.linalg.norm(unnormalized_N)
    
    def CalcDN(self, P, dP: np.ndarray) -> np.ndarray:
        dir_vec = self._P2 - self._P1
        total_length_sq = np.dot(dir_vec, dir_vec)
        
        if total_length_sq == 0:
            return np.zeros(2)  # Avoid division by zero

        dT = np.dot(dP, dir_vec) / total_length_sq
        # change in unnormalized normal
        dn = (self._normal2 - self._normal1) * dT
        n = self.UnnormalizedShadingN(P)

        return (dn * np.dot(n, n) - n * np.dot(n, dn)) / (np.dot(n, n) ** 1.5)
    
    def ShapeMatrix(self, P: np.ndarray) -> np.ndarray:
        n = self.UnnormalizedShadingN(P)
        f = 1.0 / (np.dot(n, n) ** 1.5) # normalization factor
        dir_vec = self._P2 - self._P1
        const = np.outer(self._normal2 - self._normal1, dir_vec) / np.dot(dir_vec, dir_vec)
        
        return f * (np.dot(n, n) * np.identity(2) - np.outer(n, n)) @ const

    def Ior(self) -> float:
        return self._ior

    def __repr__(self) -> str:
        return f"Plane(P1={self._P1}, P2={self._P2}, N={self._normal}, IOR={self._ior})"
