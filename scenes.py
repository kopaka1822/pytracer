import numpy as np
from plane import Plane
from ray import Ray
from hit import Hit

# ---------------------------------------------------------------
# Scene setup: finite line segments
# ---------------------------------------------------------------

# adds the circle located at P with (2D) radius r.x r.y to the planes list
def addCircle(planes, P, r, faceOutside, ior, numSegments=16):
    for i in range(numSegments):
        theta1 = (i / numSegments) * 2 * np.pi
        theta2 = ((i + 1) / numSegments) * 2 * np.pi
        theta0 = ((i - 1) / numSegments) * 2 * np.pi
        theta3 = ((i + 2) / numSegments) * 2 * np.pi
        p0 = P + r * np.array([np.cos(theta0), np.sin(theta0)])
        p1 = P + r * np.array([np.cos(theta1), np.sin(theta1)])
        p2 = P + r * np.array([np.cos(theta2), np.sin(theta2)])
        p3 = P + r * np.array([np.cos(theta3), np.sin(theta3)])
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

refraction_scene = [

    # two planes in center for glass
    Plane([-10, 2], [10, 2], ior=1.5),
    Plane([10, 0], [-10, -1], ior=1.5),
]
addBox(refraction_scene, [-10, -10], [10, 10])

glass_scene = [
    #Plane([10, -10], [10, 20]), # wall on the right
    #Plane([-10, -10], [10, -10]), # wall on the bottom
]
#addBox(refraction_scene2, [-10, -10], [10, 10])
addCircle(glass_scene, [0, 3], [2, 2], faceOutside=True, ior=1.5, numSegments=32)
addCircle(glass_scene, [0, 3], [1.8, 1.8], faceOutside=False, ior=1.5, numSegments=32)

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
addCircle(glass_globe_scene, [0, 3], [3, 3], faceOutside=True, ior=1.5, numSegments=32)

# SET SCENE ---------------------------------------------- #
planes = glass_globe_scene

def closestIntersect(ray: Ray, prevPlane: Plane | None = None) -> Hit | None:
    closest_hit = None
    min_t = float('inf')

    for pl in planes:
        if pl is prevPlane:
            continue
        hit = ray.calcHit(pl)
        if hit is not None and hit.T() < min_t:
            min_t = hit.T()
            closest_hit = hit

    return closest_hit