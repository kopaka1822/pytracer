import numpy as np
from hit import Hit
from plane import Plane

class Ray:
    # static variables
    tangent_scale = 0.14
    normalized = True
    pseudo_differentials = False

    def __init__(self, P: np.ndarray, D: np.ndarray, dP: np.ndarray | None = None, dD: np.ndarray | None = None):
        self._P = np.asarray(P, dtype=float)
        D = np.asarray(D, dtype=float)
        if Ray.normalized:
            D /= np.linalg.norm(D)
        self._D = D

        self._dP = np.zeros(2) if dP is None else np.asarray(dP, dtype=float)
        # tangent based perpedicular vector
        self._Right = Ray.tangent_scale * np.array([-D[1], D[0]])
        # dD = (dot(d, d) * Right - dot(d, Right) * d) / (dot(d, d) ** 1.5)
        # here: let d = D
        if Ray.normalized:
            self._dD = (np.dot(D, D) * self._Right - np.dot(D, self._Right) * D) / (np.dot(D, D) ** 1.5) if dD is None else np.asarray(dD, dtype=float)
        else:
            self._dD = self._Right if dD is None else np.asarray(dD, dtype=float) # unnormalized case is simply the right vector

    # --- Getter ---
    def P(self) -> np.ndarray:
        return self._P

    def D(self) -> np.ndarray:
        return self._D

    def dP(self) -> np.ndarray:
        return self._dP

    def dD(self) -> np.ndarray:
        return self._dD

    # shift based on ray differential right vector
    def shiftS(self, s: float) -> "Ray":
        newD = self._D + s * self._Right
        return Ray(self._P, newD, self._dP, self._dD)

    def calcHit(self, plane: Plane, forceIntersect: bool = False) -> Hit | None:
        N = plane.N()
        P0 = plane.P()
        denom = np.dot(N, self._D)
        if abs(denom) < 1e-8:
            return None  # Parallel

        t = np.dot(N, P0 - self._P) / denom
        if t <= 0:
            return None  # intersect behind plane

        hit_point = self._P + t * self._D

        if not forceIntersect:
            # check if between P1 and P2
            seg_vec = plane.P2() - plane.P1()
            hit_vec = hit_point - plane.P1()
            proj = np.dot(hit_vec, seg_vec)
            if proj < 0 or proj > np.dot(seg_vec, seg_vec):
                return None

        return Hit(plane, hit_point, t)

    def transfer(self, hit: Hit) -> "Ray":
        if Ray.pseudo_differentials:
            dt = - np.dot(self._dP + hit.T() * self._dD, hit.Plane().N()) / np.dot(self._D + self._dD, hit.Plane().N())
            dPNew = self._dP + hit.T() * self._dD + dt * (self._D + self._dD)
        else:
            dt = - np.dot(self._dP + hit.T() * self._dD, hit.Plane().N()) / np.dot(self._D, hit.Plane().N())
            dPNew = self._dP + hit.T() * self._dD + dt * self._D
        return Ray(hit.P(), self._D, dPNew, self._dD)

    def reflect(self, hit: Hit) -> "Ray":
        N = hit.Plane().N()
        D = self._D
        R = D - 2 * np.dot(D, N) * N
        dN = 0 # assume zero normal differentials for planes
        dDN = np.dot(self._dD, N) + np.dot(D, dN)
        dDNew = self._dD - 2 * (np.dot(D, N) * dN + dDN * N)
        #dDNew = self._dD - 2 * np.dot(self._dD, N) * N # for dN = 0
        return Ray(hit.P(), R, self._dP, dDNew)

    def _refract(self, I, N, eta):
        if np.dot(I, N) > 0:
            N = -N
        cosi = -np.dot(N, I)   # >= 0
        k = 1.0 - eta**2 * (1.0 - cosi**2)
        if k < 0:
            return None  # TIR
        T = eta * I + (eta * cosi - np.sqrt(k)) * N
        return T / np.linalg.norm(T)

    def refract(self, hit: Hit) -> "Ray | None":
        N = hit.Plane().N()
        eta = 1.0 / hit.Plane().Ior()
        D = self._D

        if np.dot(N, -D) < 0:
            # flip
            N = -N
            eta = hit.Plane().Ior()

        Dnew = self._refract(D, N, eta)
        if Dnew is None:
            return None

        # ray diff
        dN = 0
        mu = eta * np.dot(D, N) - np.dot(Dnew, N)
        dDN = np.dot(self._dD, N) + np.dot(D, dN)
        dmu = (eta - (eta * eta * np.dot(D, N) / np.dot(Dnew, N))) * dDN # Note: falcor implementaion uses eta + eta * eta instead?
        #dmu = (eta + (eta * eta * np.dot(D, N) / np.dot(Dnew, N))) * dDN # Note: falcors dot prodcut has inversed sign for Dnew => thats why its reversed.
        dDNew = eta * self._dD - (mu * dN + dmu * N)

        return Ray(hit.P(), Dnew, self._dP, dDNew)

    def sampleNext(self, hit: Hit) -> "Ray | None":
        if hit.Plane().Ior() == 1.0:
            return self.reflect(hit)
        else:
            return self.refract(hit)

    def __repr__(self):
        return f"Ray(P={self._P}, D={self._D})"
