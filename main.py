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
C1 = np.array([-8.55, 4.5])
C1_angle = 0.0
max_bounces = 10
# scene specific overwrites
if planes is reflection_scene3:
    C1 = np.array([-5.4, 3.5])
    max_bounces = 3
if planes is glass_scene:
    C1 = np.array([-7.5, 5.0])
if planes is glass_globe_scene:
    C1 = np.array([-6.4, 2.4])
    C1_angle = -10.8

draw_differentials = True
draw_guess = True
draw_normals = False
monte_carlo = False # use monte carlo sampling for refraction/reflection decisions
methods = ["PointToLight", "LightToPoint"]
# selected method for UI
selected_method = 0

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
    ax.plot(C1[0], C1[1], 'go')
    ax.text(C1[0]-0.4, C1[1]+0.2, LABEL_C1, color='g')
    # draw light
    ax.plot(L1[0], L1[1], 'ro')
    ax.text(L1[0]-0.4, L1[1]+0.2, LABEL_L1, color='r')

    # Draw C1 ray direction
    dir_len = 1.5
    dir = np.array([np.cos(np.radians(C1_angle)), np.sin(np.radians(C1_angle))])
    ax.arrow(C1[0], C1[1], dir[0]*dir_len, dir[1]*dir_len,
             head_width=0.2, color='g', length_includes_head=True)

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
        ax.plot([ray.P()[0], hit.P()[0]], [ray.P()[1], hit.P()[1]], 'g-', label=LABEL_RAY if i == 0 else None)

        # Transfer the ray to the hit point
        prevDiffP = ray.P() + ray.dP()
        ray = ray.transfer(hit)
        curDiffP = ray.P() + ray.dP()
        if draw_differentials:
            ax.plot([prevDiffP[0], curDiffP[0]], [prevDiffP[1], curDiffP[1]], 'b--', label=LABEL_RAY_DIFF if i == 0 else None)

        lastP = ray.P()
        ray = ray.sampleNext(hit, sampler)
        if ray is None:
            break  # refraction not possible
        prevPlane = hit.Plane()

    # draw point P
    ax.plot(lastP[0], lastP[1], 'go')
    ax.text(lastP[0]+LABEL_P_OFFSET, lastP[1]+0.2, LABEL_P, color='g')

    # draw direct connection from P to L1
    ax.plot([lastP[0], L1[0]], [lastP[1], L1[1]], 'r-', label="Direct Connection")
    
    ax.legend(loc="upper right")
    fig.canvas.draw_idle()

# ---------------------------------------------------------------
# Manifold Exploration Method
# ---------------------------------------------------------------

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

def methodManifoldExplore(C0, C1, dir, hits, sampler):
    if len(hits) == 0: return dir # envmap hit
    if len(hits) == 1: return hits[0].P() - C0 # direct connection
    
    # in normal ME, x1 is fixed and xn is varied. We want to vary x1 (C1->C0) and keep P fixed (xn), so we reverse the hits
    rhits = reverseHits(C0, dir, hits, includeP=True)
    rsampler = sampler.reverse()

    print("-----------------------------------------------------------------------------")
    beta = 1.0
    for i in range(max(iterations, 1)):
        dp = C0 - rhits[-1].P() # = (xn'-xn). rhits[-1] should be C1 initially (but projected onto the C0 plane)
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
        p0dir = p1new - rhits[0].P()
        rhitsnew = [rhits[0]]
        rsamplernew = rsampler.copy()

        # trace new hits
        ray2 = Ray(rhits[0].P(), p0dir)
        prevPlane = hits[-1].Plane() # plane at P
        cplane = rhits[-1].Plane() # plane at C0 (final plane)

        for j in range(1, len(rhits) - 1):
            hit2 = closestIntersect(ray2, prevPlane)
            if hit2 is None:
                break # TODO intersect with P-plane
            if draw_last_iteration and i == iterations - 1:
                ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'r-', label=LABEL_RAY2_ITERATION if j == 1 else None)
            if draw_guess and i == 0:
                ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'b-', label=LABEL_RAY2 if j == 1 else None)
            rhitsnew.append(hit2)
            ray2 = ray2.transfer(hit2)
            ray2 = ray2.sampleNext(hit2, rsamplernew)
            if ray2 is None:
                break
            prevPlane = hit2.Plane()
        
        # final intersection with C0 plane
        if ray2 is not None:
            hit2 = ray2.calcHit(cplane, forceIntersect=True)
            if hit2.T() > 0:
                if draw_last_iteration and i == iterations - 1:
                    ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'r-', label=None)
                if draw_guess and i == 0:
                    ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'b-', label=None)
                rhitsnew.append(hit2)

        foundBetter = False
        if len(rhitsnew) != len(rhits):
            print(f"ME {i+1}: expected {len(rhits)} hits, got {len(rhitsnew)} hits, reducing beta.")
        else:
            # check if error got smaller
            dpold = C0 - rhits[-1].P()
            dpnew = C0 - rhitsnew[-1].P()
            if np.linalg.norm(dpnew) < np.linalg.norm(dpold):
                rhits = rhitsnew
                print(f"ME {i+1}: improved solution with |dp|={np.linalg.norm(dpnew):.4g}.")
                beta = min(1.0, beta * 2.0)
                foundBetter = True
            else:
                print(f"ME {i+1}: no improvement (|dpold|={np.linalg.norm(dpold):.4g}, |dpnew|={np.linalg.norm(dpnew):.4g}), reducing beta.")
        
        if not foundBetter:
            beta = beta * 0.5

    newDir = rhits[-2].P() - rhits[-1].P()
    return newDir / np.linalg.norm(newDir)

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
