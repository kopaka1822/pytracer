from main import *

# mirrors scene at plane with 2D location O and 2D normal N (return 3x3 matrix)
def matrixMirror(O, N):
    nx, ny = N
    ox, oy = O
    d = nx * ox + ny * oy
    M = np.array([
        [1 - 2*nx*nx,   -2*nx*ny,    2*nx*d],
        [-2*nx*ny,      1 - 2*ny*ny, 2*ny*d],
        [0,             0,           1]
    ])
    return M

# performs a shear transformation about point O with normal N and shear factor s
def matrixShear(O, N, s):
    nx, ny = N
    ox, oy = O
    d = nx * ox + ny * oy
    M = np.array([
        [1 - s*nx*ny, -s*ny*ny,  s*ny*d],
        [ s*nx*nx, 1 + s*nx*ny, -s*nx*d],
        [0, 0, 1]
    ])
    return M

def methodRayLength(C0, C1, dir, hits):
    rayLength = 0.0
    speed = 1.0
    ray = Ray(C1, dir) # only used for tracking direction
    for hit in hits:
        rayLength += hit.T()
        if useSpeed:
            rayLength = rayLength / speed
        cosalpha = abs(np.dot(ray.D(), hit.Plane().N()))
        ray = ray.transfer(hit)
        ray = ray.sampleNext(hit)
        cosbeta = abs(np.dot(ray.D(), hit.Plane().N()))
        speed *= cosalpha / cosbeta

    newDir = C1 + dir * rayLength - C0
    draw_prediction(C0, newDir, hits)

# matrix multiplication
def mul(A, B):
    return np.matmul(A, B)

def methodReflectAndShear(C0, dir, hits):
    # initialize viewTransform with a 3x3 identity matrix
    viewTransform = np.identity(3)
    ray = Ray(C1, dir) # only used for tracking direction
    for hit in hits[:-1]:
        I = -ray.D() # incomming direction
        eta = ray.eta(hit)
        ray = ray.transfer(hit).sampleNext(hit)
        R = ray.D() # outgoing direction
        H = (I + R) / np.linalg.norm(I + R)  # half-vector
        refraction = hit.Plane().Ior() != 1.0
        if useShear and refraction:
            # compute shear factor s
            cosalpha = abs(np.dot(I, hit.Plane().N()))
            cosbeta = abs(np.dot(R, hit.Plane().N()))
            sinalpha = np.sqrt(1 - cosalpha * cosalpha)
            s = sinalpha * (cosbeta - eta * cosalpha) / (cosalpha * cosbeta)
            viewTransform = mul(viewTransform, matrixShear(hit.P(), hit.Plane().N(), s))
        else:
            viewTransform = mul(viewTransform, matrixMirror(hit.P(), H))

    P = hits[-1].P()
    Pnew = mul(viewTransform, np.array([P[0], P[1], 1.0]))[:2]

    newDir = Pnew - C0
    newDir /= np.linalg.norm(newDir)
    draw_prediction(C0, newDir, hits)