import numpy as np
from plane import Plane
from ray import Ray
from hit import Hit

# ---------------------------------------------------------------
# Scene setup: finite line segments
# ---------------------------------------------------------------

# adds the circle located at P with (2D) radius r.x r.y to the planes list
def addCircle(planes, P, r, faceOutside, ior, numSegments=128, percentage=1.0):
    for i in range(numSegments):
        theta1 = (i / numSegments) * 2 * np.pi * percentage
        theta2 = ((i + 1) / numSegments) * 2 * np.pi * percentage
        theta0 = ((i - 1) / numSegments) * 2 * np.pi * percentage
        theta3 = ((i + 2) / numSegments) * 2 * np.pi * percentage
        p0 = P - r * np.array([-np.sin(theta0), np.cos(theta0)])
        p1 = P - r * np.array([-np.sin(theta1), np.cos(theta1)])
        p2 = P - r * np.array([-np.sin(theta2), np.cos(theta2)])
        p3 = P - r * np.array([-np.sin(theta3), np.cos(theta3)])
        if not faceOutside:
            planes.append(Plane(p1, p2, ior=ior, P0=p0, P3=p3))
        else:
            planes.append(Plane(p2, p1, ior=ior, P0=p3, P3=p0))

def addBox(planes, Pmin, Pmax, faceOutside = True, ior=1.0):
    corners = [
        np.array([Pmin[0], Pmin[1]]),
        np.array([Pmax[0], Pmin[1]]),
        np.array([Pmax[0], Pmax[1]]),
        np.array([Pmin[0], Pmax[1]]),
    ]
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        if not faceOutside:
            planes.append(Plane(p1, p2, ior=ior))
        else:
            planes.append(Plane(p2, p1, ior=ior))

def addWave(planes, Pstart, Pend, amplitude, frequency, ior=1.333, numSegments=128):
    Pstart = np.array(Pstart)
    Pend = np.array(Pend)
    N = (Pend - Pstart) / np.linalg.norm(Pend - Pstart)
    N = np.array([-N[1], N[0]]) # rotate by 90 degrees to get normal
    for i in range(numSegments):
        t1 = i / numSegments
        t2 = (i + 1) / numSegments
        t0 = (i - 1) / numSegments
        t3 = (i + 2) / numSegments
        p0 = (1 - t0) * Pstart + t0 * Pend + N * amplitude * np.sin(2 * np.pi * frequency * t0)
        p1 = (1 - t1) * Pstart + t1 * Pend + N * amplitude * np.sin(2 * np.pi * frequency * t1)
        p2 = (1 - t2) * Pstart + t2 * Pend + N * amplitude * np.sin(2 * np.pi * frequency * t2)
        p3 = (1 - t3) * Pstart + t3 * Pend + N * amplitude * np.sin(2 * np.pi * frequency * t3)
        planes.append(Plane(p1, p2, ior=ior, P0=p0, P3=p3))

reflection_scene = [
    Plane([-10, -5], [10, -5]),       # bottom
    Plane([-8, -3], [-3, -1]),        # slanted left
    Plane([3, -1], [8, -3]),          # slanted right
    Plane([-5, 5], [-2, 8]),          # upper left
    Plane([2, 7], [5, 5]),            # upper right
    Plane([-1, 8], [1, 8]),           # top
]
addBox(reflection_scene, [-10, -10], [10, 10])

reflection_scene2 = [
    Plane([-3, 7], [3, 7]),
    Plane([3, 0], [7, 0]),
]
addBox(reflection_scene2, [-10, -10], [10, 10])

reflection_scene3 = [
    Plane([-5, 0], [0, 5]),
    Plane([-5, 5], [0, 10]),
    Plane([2.5, 10], [2.5, 5]),
    Plane([2.5, 5], [0, 5]),
]

refraction_scene = [

    # two planes in center for glass
    Plane([-10, 2], [10, 2], ior=1.5),
    Plane([10, 0], [-10, 0], ior=1.5),
]
addBox(refraction_scene, [-10, -10], [10, 10])

glass_scene = []
addBox(glass_scene, [-10, -10], [10, 10], ior=0.0)
addCircle(glass_scene, [0, 3], [2, 2], faceOutside=True, ior=1.5)
addCircle(glass_scene, [0, 3], [1.6, 1.6], faceOutside=False, ior=1.5)


egg_scene = []
addBox(egg_scene, [-10, -10], [10, 10], ior=0.0)
addCircle(egg_scene, [0, 3], [2, 4], faceOutside=True, ior=1.5)
addCircle(egg_scene, [0, 3], [1.6, 3.2], faceOutside=False, ior=1.5)

glasses_scene = [
    Plane([10, -40], [10, 40], ior=0.0), # wall on the right
]
addBox(glasses_scene, [-10, -10], [10, 10], ior=0.0)
# glass 1
addCircle(glasses_scene, [0, 3], [1.2, 1.2], faceOutside=True, ior=1.5)
addCircle(glasses_scene, [0, 3], [1, 1], faceOutside=False, ior=1.5)
# glass 2
addCircle(glasses_scene, [5, 1], [1.2, 1.2], faceOutside=True, ior=1.5)
addCircle(glasses_scene, [5, 1], [1, 1], faceOutside=False, ior=1.5)
# glass 3
addCircle(glasses_scene, [5, 5], [1.2, 1.2], faceOutside=True, ior=1.5)
addCircle(glasses_scene, [5, 5], [1, 1], faceOutside=False, ior=1.5)

glass_globe_scene = [
    #Plane([10, -40], [10, 40], ior=0.0), # wall on the right
    #Plane([-10, -10], [10, -10]), # wall on the bottom
]
addBox(glass_globe_scene, [-10, -10], [10, 10], ior=0.0)
addCircle(glass_globe_scene, [0, 3], [3, 3], faceOutside=True, ior=1.5, numSegments=256)

pool_scene = []
addBox(pool_scene, [-10, -10], [10, 10], ior=0.0)
addWave(pool_scene, [-10, 0], [10, 0], amplitude=0.4, frequency=10.0, numSegments=256)

lense_scene = []
addBox(lense_scene, [-10, -10], [10, 10], ior=0.0)
addCircle(lense_scene, [0, 3], [1.5, 3], faceOutside=True, ior=1.5, percentage=0.5)
lense_scene.append(Plane([0, 0], [0, 6], ior=1.5))

# SET SCENE ---------------------------------------------- #
planes = lense_scene

def closestIntersect(ray: Ray, prevPlane: Plane | None = None, TMin: float = 0.0, TMax: float = float('inf')) -> Hit | None:
    closest_hit = None
    min_t = float('inf')

    for pl in planes:
        if pl is prevPlane:
            continue
        hit = ray.calcHit(pl)
        if hit is not None and TMin < hit.T() < TMax and hit.T() < min_t:
            min_t = hit.T()
            closest_hit = hit

    return closest_hit