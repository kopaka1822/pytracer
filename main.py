import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider, RadioButtons
from plane import Plane
from ray import Ray
from hit import Hit
from scenes import *
from sampler import *
import random

# Configure matplotlib to use LaTeX rendering
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
# })


# Camera positions (modifiable via sliders)
L1 = np.array([-8.2, 5.0])
C1 = np.array([-6.8, 0.8])
C1_angle = -14.4
max_bounces = 10
# scene specific overwrites
# if planes is reflection_scene3:
#     C1 = np.array([-5.4, 3.5])
#     max_bounces = 3
if planes is glass_scene:
     C1 = np.array([-6.8, -5.33])
     C1_angle = 8.4
     L = np.array([-8.2, 5.0])
# if planes is glass_globe_scene:
#     C1 = np.array([-6.4, 2.4])
#     C1_angle = -10.8

draw_differentials = True
draw_guess = True
draw_normals = False
monte_carlo = False # use monte carlo sampling for refraction/reflection decisions
methods = ["PointToLight", "LightToPoint"]
# selected method for UI
selected_method = 1
# manifold exploration for reference
manifold_iterations = 20
draw_last_iteration = True

# new RNG seed (0..1)
rng_seed = 0

# labels
LABEL_RAY = "C ray"
LABEL_RAY2 = "L ray"
LABEL_RAY_DIFF = "ray diff"
LABEL_P = "$P$"
LABEL_P_OFFSET = 0.2 # xoffset 
LABEL_C1 = "$C$"
LABEL_L1 = "$L$"

# ---------------------------------------------------------------
# Draw functions
# ---------------------------------------------------------------

def draw_scene():
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_title(f"Caustics - {methods[selected_method]}")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Draw finite planes
    for pl in planes:
        P1, P2 = pl.P1(), pl.P2()
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 'k-', lw=1.5)
        # draw normal at midpoint
        if draw_normals:
            mid = (P1 + P2) / 2
            N = pl.N()
            ax.arrow(mid[0], mid[1], N[0]*0.8, N[1]*0.8,
                     head_width=0.2, color='r', length_includes_head=True)

    # Draw cameras
    ax.plot(C1[0], C1[1], marker='o', color='deepskyblue')
    ax.text(C1[0]-0.4, C1[1]+0.2, LABEL_C1, color='deepskyblue')
    # draw light (black)
    ax.plot(L1[0], L1[1], marker='o', color='black')
    ax.text(L1[0]-0.4, L1[1]+0.2, LABEL_L1, color='black')

    # Draw C1 ray direction
    dir_len = 1.5
    dir = np.array([np.cos(np.radians(C1_angle)), np.sin(np.radians(C1_angle))])
    ax.arrow(C1[0], C1[1], dir[0]*dir_len, dir[1]*dir_len,
             head_width=0.2, color='deepskyblue', length_includes_head=True)

    prevPlane = None
    lastP = C1
    ray = Ray(C1, dir)
    hits = []
    # main loop
    sampler = DeterministicSampler()
    if monte_carlo:
        sampler = RandomSampler(rng_seed)

    for i in range(max_bounces):
        # Find the closest intersection
        hit = closestIntersect(ray, prevPlane)
        if hit is None:
            break
        hits.append(hit)

        # Draw the ray to the hit point
        ax.plot([ray.P()[0], hit.P()[0]], [ray.P()[1], hit.P()[1]], color='deepskyblue', linestyle='-', label=LABEL_RAY if i == 0 else None)

        # Transfer the ray to the hit point
        prevDiffP = ray.P() + ray.dP()
        ray = ray.transfer(hit)
        curDiffP = ray.P() + ray.dP()
        if draw_differentials:
            ax.plot([prevDiffP[0], curDiffP[0]], [prevDiffP[1], curDiffP[1]], color='deepskyblue', linestyle='--', label=LABEL_RAY_DIFF if i == 0 else None)

        lastP = ray.P()
        lastRay = ray
        ray = ray.sampleNext(hit, sampler)
        if ray is None:
            ray = lastRay
            break  # refraction not possible
        prevPlane = hit.Plane()

    # draw point P
    ax.plot(lastP[0], lastP[1], marker='o', color='deepskyblue')
    ax.text(lastP[0]+LABEL_P_OFFSET, lastP[1]+0.2, LABEL_P, color='deepskyblue')
    
    # get direct hits from P to L1 and plot them
    phit = hits[-1]
    hits = getDirectHits(lastP, L1, prevPlane)
    draw_hits(hits, color='orange', drawRay=False, drawDiff=False)

    # draw manifold exploration result
    methodManifoldExplore(L1, lastP)

    if selected_method == 0:
        methodPointToLight(lastP, L1, hits, ray)
    if selected_method == 1:
        methodLightToPoint(lastP, L1, hits, ray, phit)

    # draw direct connection from P to L1 (orange)
    if draw_guess:
        ax.plot([lastP[0], L1[0]], [lastP[1], L1[1]], color='orange', linestyle='-', label="Direct Connection")

    

    ax.legend(loc="upper right")
    fig.canvas.draw_idle()

def draw_hits(hits, color, drawRay = True, drawDiff = True, rayLabel=None, diffLabel=None, initialdP=None, initialdD=None):
    for hit in hits:
        ax.plot(hit.P()[0], hit.P()[1], marker='o', color=color)

    if drawRay:
        for i in range(len(hits) - 1):
            ax.plot([hits[i].P()[0], hits[i+1].P()[0]], [hits[i].P()[1], hits[i+1].P()[1]], color=color, linestyle='-', label=rayLabel if i == 0 else None)

    if drawDiff and draw_differentials: # reconstruct ray to do differentials
        ray = Ray(hits[0].P(), hits[1].P() - hits[0].P(), initialdP, initialdD)
        for i in range(1, len(hits)):
            hit = hits[i]

            # transfer to hit
            startdP = ray.P() + ray.dP()
            ray = ray.transfer(hit)
            enddP = ray.P() + ray.dP()

            ax.plot([startdP[0], enddP[0]], [startdP[1], enddP[1]], color=color, linestyle='--', label=diffLabel if i == 1 else None)

            if hit.Plane().Ior() == 0.0:
                break # stop ray here
            ray = ray.refract(hit) # TODO refract or reflect?
            if ray is None:
                break

# ----------------------------------------------------------------
# Shadow Methods
# ----------------------------------------------------------------

def getDirectHits(P, L, prevPlane):
    '''Returns the list of hits along the straight ray from P to L, without P'''

    ray = Ray(P, L - P)
    hits = []
    tmin = 0.0
    tmax = np.linalg.norm(L - P)
    while True:
        hit = closestIntersect(ray, prevPlane, tmin, tmax)
        if hit is None:
            break
        hits.append(hit)
        prevPlane = hit.Plane()
        tmin = hit.T()
        
    return hits

def computeDDforLight(P, L, dPdx):
    """
    Compute the differential of the normalized direction (L-P)/|L-P|
    with respect to a differential of P (dPdx).
    """
    d = L - P
    dd = np.dot(d, d)
    dddx = -dPdx
    return (dd * dddx - np.dot(d, dddx) * d) / (dd ** 1.5)

def methodPointToLight(P, L, hits, rayIn):
    ray = Ray(P, L - P, rayIn.dP(), computeDDforLight(P, L, rayIn.dP()))

    rayRef = ray.transfer2(L, -ray.D(), np.linalg.norm(L - P))
    # draw reference ray differential
    if draw_differentials and draw_guess:
        refStart = rayIn.P() + rayIn.dP()
        refEnd = rayRef.P() + rayRef.dP()
        ax.plot([refStart[0], refEnd[0]], [refStart[1], refEnd[1]], color='orange', linestyle='--', label=None)

    lastTMin = 0.0
    rayDirIn = ray.D()
    rayDirOut = ray.D()
    for hit in hits:
        eta = hit.Plane().Ior()
        N = hit.ShadingN()
        virtualT = hit.T() - lastTMin
        # check if front face or back face hit
        if np.dot(hit.Plane().N(), ray.D()) < 0:
            # front face hit (enter medium)
            eta = 1.0 / eta # flip ior
            rayDirIn = ray.D()
            rayDirOut = Ray._refract(ray.D(), N, eta) # always possible, eta < 1.0
        else:
            # back face hit (exit medium)
            N = -N
            rayDirOut = ray.D()
            rayDirIn = Ray._refract(ray.D(), N, 1.0 / eta) # always possible, 1/eta < 1.0
            # virtualT *= np.dot(N, rayDirOut) / np.dot(N, rayDirIn)
        
        # create ray for current ray model
        startP = ray.P()
        startdP = ray.P() + ray.dP()
        ray = Ray(ray.P(), rayDirIn, ray.dP(), ray.dD())
        ray = ray.transfer2(ray.P() + virtualT * rayDirIn, N, virtualT)
        endP = ray.P()
        enddP = ray.P() + ray.dP()

        # refract with current model
        ray = ray.refract(hit)
        ray = Ray(endP, ray.D(), ray.dP(), ray.dD())

        # draw ray model from startP to endP
        ax.plot([startP[0], endP[0]], [startP[1], endP[1]], color='green', linestyle='-', label=None)
        if draw_differentials:
            ax.plot([startdP[0], enddP[0]], [startdP[1], enddP[1]], color='green', linestyle='--', label=None)

        lastTMin = hit.T() # update tmin

    # final connection to L
    finalT = np.linalg.norm(L - P) - lastTMin

    startP = ray.P()
    startdP = ray.P() + ray.dP()
    ray = Ray(ray.P(), rayDirOut, ray.dP(), ray.dD())
    ray = ray.transfer2(ray.P() + finalT * rayDirOut, -rayDirOut, finalT)
    endP = ray.P()
    enddP = ray.P() + ray.dP()

    ax.plot([startP[0], endP[0]], [startP[1], endP[1]], color='green', linestyle='-', label="Ray Model")
    if draw_differentials:
        ax.plot([startdP[0], enddP[0]], [startdP[1], enddP[1]], color='green', linestyle='--', label="Ray Model Diff")

def methodLightToPoint(P, L, hits, rayIn, phit):
    rayRef = Ray(L, P - L, None, -computeDDforLight(P, L, rayIn.dP()))
    rayRef = rayRef.transfer2(P, phit.Plane().N(), np.linalg.norm(P - L))

    # draw reference ray differential
    if draw_differentials and draw_guess:
        refStart = L
        refEnd = rayRef.P() + rayRef.dP()
        ax.plot([refStart[0], refEnd[0]], [refStart[1], refEnd[1]], color='orange', linestyle='--', label=None)

    ray = Ray(L, P - L, None, -computeDDforLight(P, L, rayIn.dP()))

    lastTMin = 0.0
    maxT = np.linalg.norm(L - P)
    rayDirIn = ray.D()
    rayDirOut = ray.D()
    for hit in reversed(hits):
        eta = hit.Plane().Ior()
        N = hit.ShadingN()
        virtualT = maxT - hit.T() - lastTMin
        # check if front face or back face hit
        if np.dot(hit.Plane().N(), ray.D()) < 0:
            # front face hit (enter medium)
            eta = 1.0 / eta # flip ior
            rayDirIn = ray.D()
            rayDirOut = Ray._refract(ray.D(), N, eta) # always possible, eta < 1.0
        else:
            # back face hit (exit medium)
            N = -N
            rayDirOut = ray.D()
            rayDirIn = Ray._refract(ray.D(), N, 1.0 / eta) # always possible, 1/eta < 1.0
            # virtualT *= np.dot(N, rayDirOut) / np.dot(N, rayDirIn)
        
        # create ray for current ray model
        startP = ray.P()
        startdP = ray.P() + ray.dP()
        ray = Ray(ray.P(), rayDirIn, ray.dP(), ray.dD())
        ray = ray.transfer2(ray.P() + virtualT * rayDirIn, N, virtualT)
        endP = ray.P()
        enddP = ray.P() + ray.dP()

        # refract with current model
        ray = ray.refract(hit)
        ray = Ray(endP, ray.D(), ray.dP(), ray.dD())

        # draw ray model from startP to endP
        ax.plot([startP[0], endP[0]], [startP[1], endP[1]], color='green', linestyle='-', label=None)
        if draw_differentials:
            ax.plot([startdP[0], enddP[0]], [startdP[1], enddP[1]], color='green', linestyle='--', label=None)

        lastTMin = maxT - hit.T() # update tmin

    # final connection to P
    finalT = maxT - lastTMin

    startP = ray.P()
    startdP = ray.P() + ray.dP()
    ray = Ray(ray.P(), rayDirOut, ray.dP(), ray.dD())
    ray = ray.transfer2(ray.P() + finalT * rayDirOut, phit.Plane().N(), finalT)
    endP = ray.P()
    enddP = ray.P() + ray.dP()

    ax.plot([startP[0], endP[0]], [startP[1], endP[1]], color='green', linestyle='-', label="Ray Model")
    if draw_differentials:
        ax.plot([startdP[0], enddP[0]], [startdP[1], enddP[1]], color='green', linestyle='--', label="Ray Model Diff")




# ---------------------------------------------------------------
# Manifold Exploration Method
# ---------------------------------------------------------------

def traceLightToPoint(P, L, dir):
    '''All hits from actually tracing a ray from L to P, including hits[0] = L and hits[-1] = P'''
    ray = Ray(L, dir)
    prevPlane = None
    hits = []
    pplane = Plane.fromNormal(P, L - P) # simple plane at P with normal facing L, used for final intersection
    lplane = Plane.fromNormal(L, P - L) 
    hits.append(Hit(lplane, L, 0.0)) # add light as first hit
    for i in range(max_bounces):
        hit = closestIntersect(ray, prevPlane)
        phit = ray.calcHit(pplane, forceIntersect=True)
        if hit is None: 
            hits.append(phit) # final intersect with P-plane and stop
            break
        if phit.T() > 0 and phit.T() <= hit.T():
            hits.append(phit) # final intersect with P-plane and stop
            break
        hits.append(hit)
        ray = ray.transfer(hit)
        if hit.Plane().Ior() == 0.0:
            ray = None # stop ray here
        else:
            ray = ray.refract(hit) # force refraction
        if ray is None:
            # ignore refraction -> direct connect to P-Plane last
            hits.append(phit)
            break
        prevPlane = hit.Plane()
    
    return hits

def computeDerivatives(hits):
    if len(hits) < 2:
        return []
    
    derivatives = []
    
    for i in range(1, len(hits) - 1):
        p_prev = hits[i-1].P()
        p_curr = hits[i].P()
        p_next = hits[i+1].P()
        
        # Compute relevant directions and a few useful projections
        wi = p_prev - p_curr
        wo = p_next - p_curr
        ili = 1.0 / np.linalg.norm(wi)
        ilo = 1.0 / np.linalg.norm(wo)
        wi = wi * ili # normalize
        wo = wo * ilo
        
        # Surface properties
        eta = hits[i].Plane().Ior()
        n = hits[i].ShadingN()
        N = hits[i].Plane().N() # geometric normal
        dpdu = hits[i].Tangent() # geometric tangent
        dndu = hits[i].CalcDN(dpdu)
        
        # Determine eta based on ray direction and surface normal
        if np.dot(N, wi) < 0:
            # flip normals if looking from below
            N = -N
            n = -n
            dndu = -dndu
        else: # dot(N, wi) > 0
            eta = 1.0 / eta # flip eta (n1 = 1, n2 = IOR)
        
        # Half-vector (generalized for refraction)
        H = wi + eta * wo
        ilh = 1.0 / np.linalg.norm(H)
        H = H * ilh # normalize
        
        # Useful projections
        dot_H_n = np.dot(n, H)
        dot_H_dndu = np.dot(dndu, H)
        dot_u_n = np.dot(dpdu, n)
        
        # Local shading tangent frame
        s = dpdu - dot_u_n * n # in 2D: same as rotating n by 90 degrees
        ilo = ilo * eta * ilh
        ili = ili * ilh
        
        # Derivatives of C with respect to x_{i-1} 
        dH_du = (hits[i-1].Tangent() - wi * np.dot(wi, hits[i-1].Tangent())) * ili
        dH_du = dH_du - H * np.dot(dH_du, H)
        
        A = np.array([
            [np.dot(dH_du, s)]
        ]) # in 2D 1x1 matrix, in 3D would be 2x2
        
        # Derivatives of C with respect to x_i
        dH_du = -dpdu * (ili + ilo) + wi * (np.dot(wi, dpdu) * ili) + wo * (np.dot(wo, dpdu) * ilo)
        dH_du = dH_du - H * np.dot(dH_du, H)
        
        B = np.array([
            [np.dot(dH_du, s) - np.dot(dpdu, dndu) * dot_H_n - dot_u_n * dot_H_dndu]
        ]) # in 2D 1x1 matrix, in 3D would be 2x2
        
        # Derivatives of C with respect to x_{i+1} 
        dH_du = (hits[i+1].Tangent() - wo * np.dot(wo, hits[i+1].Tangent())) * ilo
        dH_du = dH_du - H * np.dot(dH_du, H)
        
        C = np.array([
            [np.dot(dH_du, s)]
        ]) # in 2D 1x1 matrix, in 3D would be 2x2
        
        derivatives.append((A, B, C))
    
    # compute A, Ainv and Bn for later:
    Bn = np.zeros((len(derivatives), 1))
    Bn[-1][0] = derivatives[-1][2][0,0] # C matrix from the last derivative set

    A = np.zeros((len(derivatives), len(derivatives)))

    for row in range(len(derivatives)):
        if row > 0: A[row, row - 1] = derivatives[row][0][0,0] # A
        A[row, row] = derivatives[row][1][0,0] # B
        if row + 1 < len(derivatives): A[row, row + 1] = derivatives[row][2][0,0] # C

    Ainv = np.linalg.inv(A)

    #print(f"ME Derivatives: A=\n{A}, Ainv=\n{Ainv}, Bn={Bn}")
    
    return Ainv, Bn

def methodManifoldExplore(L, P):
    #if len(hits) == 0: return dir # envmap hit
    #if len(hits) == 1: return hits[0].P() - C0 # direct connection
    
    # in normal ME, x1 is fixed and xn is varied. x1 = L, xn = P
    rhits = traceLightToPoint(P, L, P - L) # actual ray via direct connection from L to P

    print("-----------------------------------------------------------------------------")
    beta = 1.0
    for i in range(max(manifold_iterations, 1)):
        dp = P - rhits[-1].P() # = (xn'-xn). rhits[-1] should be close to P initialially, and converge toward P
        dp = dp.reshape((2,1)) # dim: 2x1
        Tp1 = rhits[1].Plane().Tangent().reshape((2,1)) # = T(x2) dim: 2x1
        TpnT = rhits[-1].Plane().Tangent().reshape((1,2)) # = T(xn)^T dim: 1x2
        P1 = np.zeros(len(rhits) - 2) # = P2: dim: 1xn
        P1[0] = 1.0 # only extract the second vertex (which is the first entry in the A matrix)
        P1 = P1.reshape((1, len(rhits) - 2)) # dim: 1xn
        # TODO this could be cached, only required if rhits changes
        Ainv, Bn = computeDerivatives(rhits) # Ainv: dim: nxn, Bn: dim: nx1

        # intermediate results
        tangentOffsetN = TpnT @ dp # dim: 1x1
        tangentOffset1 = P1 @ Ainv @ Bn @ tangentOffsetN # dim: 1x1
        offsetVector1 = Tp1 @ tangentOffset1 # dim: 2x1
        print(f"ME {i+1}: dp={dp.flatten()}, tangentOffsetN={tangentOffsetN.flatten()}, tangentOffset1={tangentOffset1.flatten()}, offsetVector1={offsetVector1.flatten()}, beta={beta:.4g}")

        p1new = rhits[1].P() - beta * offsetVector1.flatten() # why does wenzel use - ?
        p0dir = p1new - rhits[0].P() # direction from L (x1) to x2
        rhitsnew = [rhits[0]]

        # trace new hits
        rhitsnew = traceLightToPoint(P, L, p0dir)
        if draw_last_iteration and i == manifold_iterations - 1:
            pass
        if draw_guess and i == 0:
            pass
        
        foundBetter = False
        
        # check if error got smaller
        # TODO change this to angle error between P and L
        dpold = P - rhits[-1].P()
        dpnew = P - rhitsnew[-1].P()
        if np.linalg.norm(dpnew) < np.linalg.norm(dpold):
            rhits = rhitsnew
            print(f"ME {i+1}: improved solution with |dp|={np.linalg.norm(dpnew):.4g}.")
            beta = min(1.0, beta * 2.0)
            foundBetter = True
        else:
            print(f"ME {i+1}: no improvement (|dpold|={np.linalg.norm(dpold):.4g}, |dpnew|={np.linalg.norm(dpnew):.4g}), reducing beta.")
        
        if not foundBetter:
            beta = beta * 0.5

    draw_hits(rhits, color='red', rayLabel="ME Ray", diffLabel="ME Ray Diff")
    #newDir = rhits[-2].P() - rhits[-1].P()
    #finalHits  traceLightToPoint(P, L, newDir)
    #return newDir / np.linalg.norm(newDir)

# matrix multiplication
def mul(A, B):
    return np.matmul(A, B)

# ---------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 6))
fig.canvas.manager.set_window_title('Pytracer')
ax = fig.add_axes([0.07, 0.1, 0.6, 0.8])  # main plot area (left)

# Slider & control panel (pruned to used widgets)
# Layout: stacked controls on the right side
ax_sliders = [
    fig.add_axes([0.75, 0.95, 0.2, 0.03]),  # 0 C1.x
    fig.add_axes([0.75, 0.91, 0.2, 0.03]),  # 1 C1.y
    fig.add_axes([0.75, 0.87, 0.2, 0.03]),  # 2 C1.angle
    fig.add_axes([0.75, 0.83, 0.2, 0.03]),  # 3 L1.x
    fig.add_axes([0.75, 0.79, 0.2, 0.03]),  # 4 L1.y
    fig.add_axes([0.75, 0.75, 0.2, 0.03]),  # 5 Max Bounces
    fig.add_axes([0.75, 0.71, 0.2, 0.03]),  # 6 Draw Guess (checkbox)
    fig.add_axes([0.75, 0.67, 0.2, 0.03]),  # 7 Draw Differentials (checkbox)
    fig.add_axes([0.75, 0.63, 0.2, 0.03]),  # 8 Differential Scale (slider)
    fig.add_axes([0.75, 0.59, 0.2, 0.03]),  # 9 Draw Normals (checkbox)
    fig.add_axes([0.75, 0.48, 0.2, 0.08]),  # 10 Method (radio) - moved slightly down
    fig.add_axes([0.75, 0.42, 0.2, 0.03]),  # 11 Use N. Diff (checkbox)
    fig.add_axes([0.75, 0.38, 0.2, 0.03]),  # 12 Monte Carlo (checkbox)
    fig.add_axes([0.75, 0.34, 0.2, 0.03]),  # 13 RNG Seed (slider)
]

slider_C1x = Slider(ax_sliders[0], "C1.x", -10.0, 10.0, valinit=C1[0])
slider_C1y = Slider(ax_sliders[1], "C1.y", -10.0, 10.0, valinit=C1[1])
slider_C1a = Slider(ax_sliders[2], "C1.angle", -180.0, 180.0, valinit=C1_angle)
slider_C0x = Slider(ax_sliders[3], "L1.x", -10.0, 10.0, valinit=L1[0])
slider_C0y = Slider(ax_sliders[4], "L1.y", -10.0, 10.0, valinit=L1[1])
# integer slider for max bounces
slider_max_bounces = Slider(ax_sliders[5], "Max Bounces", 1, 10, valinit=max_bounces, valstep=1)

# checkboxes (in ascending axis order)
checkbox_draw_guess = CheckButtons(ax_sliders[6], ["Draw Guess"], [draw_guess])
checkbox_draw_differentials = CheckButtons(ax_sliders[7], ["Draw Differentials"], [draw_differentials])

# differential scale
slider_tangent_scale = Slider(ax_sliders[8], "Differential Scale", 0.001, 0.5, valinit=Ray.tangent_scale)

# draw normals
checkbox_draw_normals = CheckButtons(ax_sliders[9], ["Draw Normals"], [draw_normals])

# method selection (radio)
radio_methods = RadioButtons(ax_sliders[10], methods, active=selected_method)

# other toggles
checkbox_use_n_differentials = CheckButtons(ax_sliders[11], ["Use N Diff."], [Ray.use_normal_differential])
checkbox_monte_carlo = CheckButtons(ax_sliders[12], ["Monte Carlo refr."], [monte_carlo])

# RNG seed slider
slider_rng_seed = Slider(ax_sliders[13], "RNG Seed", 0, 100, valinit=rng_seed, valstep=1)

# ---------------------------------------------------------------
# Slider callbacks
# ---------------------------------------------------------------

def update(val):
    global C1, C1_angle, L1, max_bounces, draw_differentials, draw_guess, draw_normals, monte_carlo, rng_seed, selected_method
    C1[0] = slider_C1x.val
    C1[1] = slider_C1y.val
    C1_angle = slider_C1a.val
    L1[0] = slider_C0x.val
    L1[1] = slider_C0y.val
    max_bounces = int(slider_max_bounces.val)
    draw_guess = checkbox_draw_guess.get_status()[0]
    draw_differentials = checkbox_draw_differentials.get_status()[0]
    Ray.tangent_scale = slider_tangent_scale.val
    draw_normals = checkbox_draw_normals.get_status()[0]
    Ray.use_normal_differential = checkbox_use_n_differentials.get_status()[0]
    monte_carlo = checkbox_monte_carlo.get_status()[0]
    rng_seed = int(slider_rng_seed.val)
    selected_method = methods.index(radio_methods.value_selected)

    draw_scene()

for s in [slider_C1x, slider_C1y, slider_C1a, slider_C0x, slider_C0y, slider_max_bounces, slider_tangent_scale, slider_rng_seed]:
    s.on_changed(update)

for c in [checkbox_draw_differentials, checkbox_draw_guess, checkbox_draw_normals, checkbox_monte_carlo, checkbox_use_n_differentials, radio_methods]:
    c.on_clicked(update)

# Initial draw
draw_scene()
plt.show()
