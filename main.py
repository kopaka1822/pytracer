import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider, RadioButtons
from plane import Plane
from ray import Ray
from hit import Hit
from scenes import *

# solves A * s = B for scalar s
def solveLinearEq(A: np.ndarray, B: np.ndarray) -> float:
    if abs(A[0]) > abs(A[1]):
        s = B[0] / A[0]
    else:
        s = B[1] / A[1]
    return s


# Camera positions (modifiable via sliders)
C0 = np.array([-9.9, 2.15])
C1 = np.array([-8.55, 4.5])
#C1_angle = -46.8  # in degrees
C1_angle = 0.0
max_bounces = 1
draw_differentials = True
draw_guess = True
draw_normals = False
iterations = 1
iteration_strategies = ["Virtual iterations", "Real iterations", "Reverse real it."]
iteration_strategy = 1  # index into iteration_strategies
predict_strategies = ["ray diff", "reverse ray diff", "manifold explore", "reflect and shear"]
predict_strategy = 0  # index into predict_strategies
useSpeed = False
useShear = False
draw_last_iteration = True
EXTRA_BOUNCES = 4 # allowed number of extra bounces during real iterations

# labels
LABEL_RAY = "original"
LABEL_RAY2 = "guess"
LABEL_RAY2_DIFF = "differential"
LABEL_RAY2_PRED = "predicted"
LABEL_RAY2_ITERATION = "last iteration"

# ---------------------------------------------------------------
# Draw functions
# ---------------------------------------------------------------

def draw_scene():
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_title("Ray Differential Motion Vector")
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
    ax.plot(C0[0], C0[1], 'bo')
    ax.text(C0[0]-0.4, C0[1]+0.2, "C0", color='b')
    ax.plot(C1[0], C1[1], 'go')
    ax.text(C1[0]-0.4, C1[1]+0.2, "C1", color='g')

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
    for i in range(max_bounces):
        # Find the closest intersection
        hit = closestIntersect(ray, prevPlane)
        if hit is None:
            break
        hits.append(hit)

        # Draw the ray to the hit point
        ax.plot([ray.P()[0], hit.P()[0]], [ray.P()[1], hit.P()[1]], 'g-', label=LABEL_RAY if i == 0 else None)

        # Transfer the ray to the hit point
        ray = ray.transfer(hit)
        lastP = ray.P()
        ray = ray.sampleNext(hit)
        if ray is None:
            break  # refraction not possible
        prevPlane = hit.Plane()

    # draw point P
    ax.plot(lastP[0], lastP[1], 'go')
    ax.text(lastP[0]+0.2, lastP[1]+0.2, "P", color='g')

    if predict_strategy == 0:
        newDir = methodRayDiff(C0, dir, hits)
    if predict_strategy == 1:
        newDir = methodReverseRayDiff(C0, C1, dir, hits)
    if predict_strategy == 2:
        newDir = methodManifoldExplore(C0, C1, dir, hits)
        #newDir = methodRayLength(C0, C1, dir, hits)
    if predict_strategy == 3:
        newDir = methodReflectAndShear(C0, dir, hits)
    
    #if iteration_strategy == 1:
    # always draw actual path
    trace_and_draw_actual(C0, newDir, hits)
    #else: 
    #    draw_prediction(C0, newDir, hits)

    ax.legend(loc="upper right")
    fig.canvas.draw_idle()

# ---------------------------------------------------------------
# Ray Differential Method
# ---------------------------------------------------------------

def methodRayDiff(C0, dir, hits):
    ray2 = Ray(C0, dir)
    lastS = 0.0 # last solution for ray2 differential
    for hit in hits:
        # intersect ray2 with the same plane
        hit2 = ray2.calcHit(hit.Plane(), forceIntersect=True)
        assert hit2 is not None
        hit2.overwriteShadingN(hit.ShadingN()) # force the same shading normal to ensure that refraction/reflection is the same
        if draw_guess:
            ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'b-', label=LABEL_RAY2 if hit == hits[0] else None)
        # update ray2
        prevP = ray2.P() + ray2.dP()
        ray2 = ray2.transfer(hit2)
        curP = ray2.P() + ray2.dP()
        # draw ray differential segment
        if draw_differentials and iterations == 0:
            ax.plot([prevP[0], curP[0]], [prevP[1], curP[1]], 'b--', label=LABEL_RAY2_DIFF if hit == hits[0] else None)

        # calc current solution for ray2 differential (PStar + s * dP = P <=> s * dP = P - PStar)
        P = hit.P()
        PStar = ray2.P()
        dP = ray2.dP()
        s = solveLinearEq(dP, P - PStar)
        lastS = s

        nextRay = ray2.sampleNext(hit2)
        if nextRay is not None and hit != hits[-1]: # update ray2 except for last hit
            ray2 = nextRay

    # draw P* and differentials
    if draw_guess:
        ax.plot(ray2.P()[0], ray2.P()[1], 'bo')
        ax.text(ray2.P()[0]+0.2, ray2.P()[1]+0.2, "P*", color='b')
        if draw_differentials and iterations == 0:
            dpend = ray2.P() + ray2.dP()
            ax.arrow(ray2.P()[0], ray2.P()[1], ray2.dP()[0], ray2.dP()[1], head_width=0.4, color='c', length_includes_head=True)
            ax.plot([ray2.P()[0], dpend[0]], [ray2.P()[1], dpend[1]], 'c-', lw=2.0)
            dphalf = ray2.P() + ray2.dP() * 0.5
            ax.text(dphalf[0]+0.2, dphalf[1]+0.2, "dP", color='c')
        # draw s
        ax.arrow(ray2.P()[0], ray2.P()[1], lastS * ray2.dP()[0], lastS * ray2.dP()[1], head_width=0.3, color='m', length_includes_head=True)
        send = ray2.P() + lastS * ray2.dP()
        ax.plot([ray2.P()[0], send[0]], [ray2.P()[1], send[1]], 'm-', lw=1.0)
        shalf = ray2.P() + lastS * ray2.dP() * 0.5
        ax.text(shalf[0]+0.2, shalf[1]+0.2, "s", color='m')

    # use lastS to determine the new ray2 initial direction.
    initial_ray = Ray(C0, dir)
    newDir = initial_ray.shiftS(lastS).D()
    if iteration_strategy == 0:
        newDir = doVirtualIterations(C0, newDir, hits)
    if iteration_strategy == 1:
        newDir = doRealIterations(C0, dir, newDir, hits)
    if iteration_strategy == 2:
        newDir = doReverseRealIterations(C0, dir, newDir, hits, initialDir0=-(ray2.D() + lastS * ray2.dD()))

    newDir /= np.linalg.norm(newDir)
    return newDir

def doRealIterations(C0, dir, newDir, hits):
    if len(hits) == 0:
        return newDir
    P = hits[-1].P()
    # normal of the final intersection plane at P
    N = -dir
    if len(hits) >= 2:
        N = hits[-2].P() - hits[-1].P()
    
    PPlane = Plane(P, P + [-N[1], N[0]])  # create plane orthogonal to dir at P
    # geometric normal of the last plane
    lastPlaneN = hits[-1].Plane().N() * (-1 if np.dot(hits[-1].Plane().N(), N) < 0 else 1)

    bestDiff = float('inf')
    bestDir = newDir.copy()
    nextS = 0.0
    fails = 0 # number of iterations without improvement

    onlyPositiveMultiplier = True # check if only positive multipliers have been used
    stepMultiplier = 1.0
    print("-----------------------------------------------------------------------------")
    # refine newDir over multiple iterations
    for i in range(iterations):
        ray2 = Ray(C0, bestDir)
        if fails > 0:
            #exponent = ((-1) ** (fails)) * ((fails + 1) // 2)
            if Ray.use_normal_differential:
                #stepMultiplier = pow(2.0, -fails)
                stepMultiplier = stepMultiplier * 0.5
            else:
                sign = (-1) ** (fails+1)
                exponent = -((fails + 1) // 2)
                stepMultiplier = sign * pow(2.0, exponent)
            print(f"It: {i} no improvement (diff={lastBestDiff:.4g}, steps={lastNumSteps}), trying stepMultiplier={stepMultiplier}")
        else:
            if Ray.use_normal_differential:
                stepMultiplier = min(1.0, stepMultiplier * 2.0) # try to increase step size again (like bitterli)
            else:
                stepMultiplier = 1.0

        ray2 = ray2.shiftS(nextS * stepMultiplier) # proposed s value based on best direction
        initial_dir = ray2.D().copy() # initial direction for this iteration
        prevPlane = None
        foundBetter = False
        lastBestDiff = float('inf')
        lastNumSteps = 0

        # trace similar number of bounces as original ray
        for iHit in range(len(hits) + EXTRA_BOUNCES):
            phit = ray2.calcHit(PPlane, forceIntersect=True)
            hit = closestIntersect(ray2, prevPlane)

            # last hit or T>0 and front face hit
            testPPlane = (phit.T() > 0) and (np.dot(N, ray2.D()) < 0)

            if testPPlane:
                PStar = phit.P()
                diff = np.linalg.norm(P - PStar)**2
                if hit is not None:
                    #diff += 10.0 * np.linalg.norm(PStar - hit.P())**2
                    diff += max(0.0, np.dot(lastPlaneN, hit.P() - P))**2 # penalize if in front of geometric plane (but not on or behind)
                throughputPenalty = True
                if throughputPenalty:
                    diff *= (1 + 0.1 * abs(len(hits) - (iHit + 1))) # larger errors if throughput / path length differs
                    #diff += abs(len(hits) - (iHit + 1)) # larger errors if throughput / path length differs

                if diff < lastBestDiff: # for logging
                    lastBestDiff = diff
                    lastNumSteps = iHit + 1

                if diff < bestDiff:
                    bestDiff = diff
                    bestDir = initial_dir.copy()

                    # calc current solution for ray2 differential (PStar + s * dP = P <=> s * dP = P - PStar)
                    dP = ray2.transfer(phit).dP()
                    nextS = solveLinearEq(dP, P - PStar)
                    foundBetter = True

            if hit is None:
                break # finished with this iteration

            prevPlane = hit.Plane()
            ray2 = ray2.transfer(hit)
            ray2 = ray2.sampleNext(hit)
            if ray2 is None: break # refraction not possible 

        if not foundBetter:
            fails += 1
            if (i + 1) == iterations:
                print(f"It: {i+1} no improvement (diff={lastBestDiff:.4g}, steps={lastNumSteps})")
        else:
            if onlyPositiveMultiplier and stepMultiplier < 0:
                onlyPositiveMultiplier = False
            print(f"It: {i+1} found better solution with diff={bestDiff:.4g}, steps={lastNumSteps}, nextS={nextS:.4g} using stepMultiplier={stepMultiplier}.")
            fails = 0

        if draw_last_iteration and (i + 1) == iterations:
            trace_and_draw_actual(C0, initial_dir, hits, color='red', label=LABEL_RAY2_ITERATION)

    if not onlyPositiveMultiplier:
        print("WARNING: negative step multipliers were used during real iterations.")

    return bestDir

def doVirtualIterations(C0, newDir, hits):
    P = hits[-1].P()
    bestDiff = float('inf')
    bestDir = newDir.copy()

    # refine newDir over multiple iterations
    for i in range(iterations):
        ray2 = Ray(C0, newDir)
        # trace ray through all hits
        for hit in hits:
            hit2 = ray2.calcHit(hit.Plane(), forceIntersect=True)
            hit2.overwriteShadingN(hit.ShadingN()) # force the same shading normal to ensure that refraction/reflection is the same (for virtual iterations)
            if draw_last_iteration and (i + 1) == iterations:
                ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'r-', label=LABEL_RAY2_ITERATION if hit == hits[0] else None)

            prevP = ray2.P() + ray2.dP() # for differential
            ray2 = ray2.transfer(hit2)
            curP = ray2.P() + ray2.dP() # for differential
            if draw_last_iteration and (i + 1) == iterations and draw_differentials:
                ax.plot([prevP[0], curP[0]], [prevP[1], curP[1]], 'r--', label=LABEL_RAY2_DIFF if hit == hits[0] else None)

            nextRay = ray2.sampleNext(hit2)
            if nextRay is not None:
                ray2 = nextRay
        
        # check final position
        curP = ray2.P()
        diff = np.linalg.norm(P - curP)
        if diff < bestDiff:
            bestDiff = diff
            bestDir = newDir.copy()

            # calc current solution for ray2 differential (PStar + s * dP = P <=> s * dP = P - PStar)
            PStar = ray2.P()
            dP = ray2.dP()
            s = solveLinearEq(dP, P - PStar)
            
            # update newDir
            newDir = Ray(C0, newDir).shiftS(s).D()
    
    return bestDir

def doReverseRealIterations(C0, dir, newDir, hits, initialDir0 = None):
    print("-----------------------------------------------------------------------------")
    if iterations <= 0: return newDir # do nothing
    if len(hits) == 0:
        return None
    P = hits[-1].P()
    d0Start = -dir
    if len(hits) >= 2:
        d0Start = hits[-2].P() - hits[-1].P()
    d0Start /= np.linalg.norm(d0Start) # normalize for computations below

    # camera plane at C0
    CPlane = Plane(C0, C0 + [dir[1], -dir[0]])  # create plane orthogonal to dir at C0
    assert np.dot(CPlane.N(), dir) > 0.0 # make sure normal points "forward"

    bestDiff = float('inf')
    bestDir0 = d0Start # direction from P to C0
    bestDirN = dir # direction from C0 to P
    nextS = 0 # actually, just repeat the d0Start ray to determine the error
    if initialDir0 is not None:
        # solve (d0Start - t * intialDirN) * d0Start = 0 => t = (d0Start * d0Start) / (intialDirN * d0Start)
        initialDir0 /= np.linalg.norm(initialDir0)
        t = np.dot(d0Start, d0Start) / np.dot(initialDir0, d0Start)
        ddWanted = t * initialDir0 - d0Start
        nextS = solveLinearEq(Ray(P, bestDir0).dD(), ddWanted)
        print(f"Using initialDir0 to determine starting nextS = {nextS:.4g}, d0Start = {d0Start}, initialDir0 = {initialDir0}, ddWanted = {ddWanted}")
    fails = 0 # number of iterations without improvement

    onlyPositiveMultiplier = True # check if only positive multipliers have been used
    stepMultiplier = 1.0
    
    # refine newDir over multiple iterations
    for i in range(iterations): # do one extra iteration for the step that we could already determine via reverse
        ray2 = Ray(P, bestDir0)
        if fails > 0:
            if Ray.use_normal_differential:
                stepMultiplier = stepMultiplier * 0.5
            else:
                sign = (-1) ** (fails+1)
                exponent = -((fails + 1) // 2)
                stepMultiplier = sign * pow(2.0, exponent)
            print(f"It: {i} no improvement (diff={lastBestDiff:.4g}, steps={lastNumSteps}), trying stepMultiplier={stepMultiplier}")
        else:
            if Ray.use_normal_differential:
                stepMultiplier = min(1.0, stepMultiplier * 2.0) # try to increase step size again (like bitterli)
            else:
                stepMultiplier = 1.0

        ray2 = ray2.shiftS(nextS * stepMultiplier) # proposed s value based on best direction
        initial_dir = ray2.D().copy() # initial direction for this iteration
        prevPlane = hits[-1].Plane() # its very important to ignore the plane we are starting on
        foundBetter = False
        lastBestDiff = float('inf')
        lastNumSteps = 0

        # trace similar number of bounces as original ray
        for iHit in range(len(hits) + EXTRA_BOUNCES):
            chit = ray2.calcHit(CPlane, forceIntersect=True)
            hit = closestIntersect(ray2, prevPlane)

            # last hit or T>0 and front face hit
            testCPlane = (chit.T() > 0) and (np.dot(CPlane.N(), ray2.D()) < 0)

            if testCPlane:
                CStar = chit.P()
                diff = np.linalg.norm(C0 - CStar)**2
                if hit is not None: # TODO penalie this even more, since we shouldnt hit anything before hitting the camera
                    diff += max(0.0, np.dot(CPlane.N(), hit.P() - C0))**2 # penalize if in front of camera plane (but not on or behind)
                throughputPenalty = True
                if throughputPenalty:
                    diff *= (1 + 0.1 * abs(len(hits) - (iHit + 1))) # larger errors if throughput / path length differs

                if diff < lastBestDiff: # for logging
                    lastBestDiff = diff
                    lastNumSteps = iHit + 1

                if diff < bestDiff:
                    bestDiff = diff
                    bestDir0 = initial_dir.copy()
                    bestDirN = -ray2.D().copy()

                    # calc current solution for ray2 differential (CStar + s * dP = P <=> s * dP = P - CStar)
                    dC = ray2.transfer(chit).dP()
                    nextS = solveLinearEq(dC, C0 - CStar)
                    foundBetter = True

            if hit is None:
                break # finished with this iteration

            prevPlane = hit.Plane()
            ray2 = ray2.transfer(hit)
            ray2 = ray2.sampleNext(hit)
            if ray2 is None: break # refraction not possible 

        if not foundBetter:
            fails += 1
            if (i + 1) == iterations and i >= 0:
                print(f"It: {i+1} no improvement (diff={lastBestDiff:.4g}, steps={lastNumSteps})")
        else:
            if onlyPositiveMultiplier and stepMultiplier < 0:
                onlyPositiveMultiplier = False
            print(f"It: {i+1} found better solution with diff={bestDiff:.4g}, steps={lastNumSteps}, nextS={nextS:.4g} using stepMultiplier={stepMultiplier}.")
            fails = 0

        if draw_last_iteration and (i + 1) == iterations:
            rhits = reverseHits(C0, dir, hits)
            trace_and_draw_actual(P, initial_dir, rhits, color='red', label=LABEL_RAY2_ITERATION, prevPlane=hits[-1].Plane())

    if not onlyPositiveMultiplier:
        print("WARNING: negative step multipliers were used during real iterations.")

    return bestDirN

# ---------------------------------------------------------------
# Reverse Ray Differential Method
# ---------------------------------------------------------------

def transferRRDiff(Jpp, Jpd, Jdp, Jdd, D, N, t): # N = geometric normal
    I = np.identity(2)
    DNTDN = np.outer(D, N) / np.dot(D, N)
    Lpp = I - DNTDN
    Lpd = t * I - t * DNTDN
    Jpp_new = mul(Jpp, Lpp)
    Jpd_new = mul(Jpp, Lpd) + Jpd
    Jdp_new = mul(Jdp, Lpp)
    Jdd_new = mul(Jdp, Lpd) + Jdd
    print(f"TransferRRDiff: t={t:.4g}, Lpp=\n{Lpp}, Lpd=\n{Lpd}")
    return Jpp_new, Jpd_new, Jdp_new, Jdd_new

def reflectRRDiff(Jpp, Jpd, Jdp, Jdd, D, N, M): # N = shading normal
    I = np.identity(2)
    Ldd = I - 2 * np.outer(N, N)
    Ldp = -2 * (np.dot(D, N) * M + mul(np.outer(N, D), M))
    Jpp_new = Jpp + mul(Jpd, Ldp)
    Jpd_new = mul(Jpd, Ldd)
    Jdp_new = Jdp + mul(Jdd, Ldp)
    Jdd_new = mul(Jdd, Ldd)
    print(f"ReflectRRDiff: Ldd=\n{Ldd}, Ldp=\n{Ldp}")
    return Jpp_new, Jpd_new, Jdp_new, Jdd_new

def refractRRDiff(Jpp, Jpd, Jdp, Jdd, D, R, N, eta, M): # N = shading normal
    I = np.identity(2)
    mu = eta * np.dot(D, N) - np.dot(R, N)
    k = eta - eta * eta * np.dot(D, N) / np.dot(R, N)
    Ldd = eta * I - k * np.outer(N, N)
    Ldp = -mu * M - k * mul(np.outer(N, D), M)
    Jpp_new = Jpp + mul(Jpd, Ldp)
    Jpd_new = mul(Jpd, Ldd)
    Jdp_new = Jdp + mul(Jdd, Ldp)
    Jdd_new = mul(Jdd, Ldd)
    print(f"RefractRRDiff: mu={mu:.4g}, k={k:.4g}, Ldd=\n{Ldd}, Ldp=\n{Ldp}")
    return Jpp_new, Jpd_new, Jdp_new, Jdd_new

def solveForDD0(Jdp, dPn, D0):
    A = Jdp.copy()
    b = dPn.copy()
    firstRowSq = A[0,0] * A[0, 0] + A[0,1] * A[0,1]
    secondRowSq = A[1,0] * A[1,0] + A[1,1] * A[1,1]
    det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
    print(f"SolveForDD0: Jdp=\n{Jdp} (det={det:.4g}), dPn={dPn}, firstRowSq={firstRowSq:.4g}, secondRowSq={secondRowSq:.4g}")
    if firstRowSq > secondRowSq: # replace second row
        A[1,0] = D0[0]
        A[1,1] = D0[1]
        b[1] = 0.0
    else: # replace first row
        A[0,0] = D0[0]
        A[0,1] = D0[1]
        b[0] = 0.0

    #dD0 = np.linalg.solve(A, b)
    invA = np.linalg.inv(A)
    det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
    print(f"SolveForDD0: Modified A=\n{A} (det={det:.4g}), b={b}, invA=\n{invA}")
    dD0 = mul(invA, b)
    return dD0

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
        
        # Get the relative index of refraction at this vertex
        eta = 1.0
        if hits[i].Plane().Ior() != 1.0:
            # Determine eta based on ray direction and surface normal
            N = hits[i].Plane().N()
            if np.dot(N, wi) < 0:
                eta = hits[i].Plane().Ior()
            else:
                eta = 1.0 / hits[i].Plane().Ior()
        
        # Half-vector (generalized for refraction)
        H = wi + eta * wo
        ilh = 1.0 / np.linalg.norm(H)
        H = H * ilh # normalize
        
        # Get surface properties
        n = hits[i].ShadingN()
        
        # tangent u (in 3D we would need two tangents, in 2D just one)
        u = hits[i].Tangent()
        dndu = hits[i].Plane().CalcDN(p_curr, u)
        dpdu = u # we dont have textcoords here, so moving one unit on the uv tangent is one unit in space
        
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
    Bn = np.zeros(len(derivatives))
    Bn[-1] = derivatives[-1][2][0,0] # C matrix from the last derivative set

    A = np.zeros((len(derivatives), len(derivatives)))

    for row in range(len(derivatives)):
        if row > 0: A[row, row - 1] = derivatives[row][0][0,0] # A
        A[row, row] = derivatives[row][1][0,0] # B
        if row + 1 < len(derivatives): A[row, row + 1] = derivatives[row][2][0,0] # C

    Ainv = np.linalg.inv(A)

    print(f"ME Derivatives: A=\n{A}, Ainv=\n{Ainv}, Bn={Bn}")
    
    return Ainv, Bn

def methodManifoldExplore(C0, C1, dir, hits):
    if len(hits) < 2:
        return dir
    
    # in normal ME, x1 is fixed and xn is varied. We want to vary x1 (C1->C0) and keep P fixed (xn), so we reverse the hits
    rhits = reverseHits(C0, dir, hits, includeP=True)

    beta = 1.0
    for i in range(iterations + 1):
        dp = C0 - rhits[-1].P() # = (xn'-xn). rhits[-1] should be C1 initially (but projected onto the C0 plane)
        Tp1 = rhits[1].Plane().Tangent() # = T(x2) dim: 2x1
        Tpn = rhits[-1].Plane().Tangent() # = T(xn) dim: 2x1
        P1 = np.zeros(len(rhits) - 1) # = P2: dim: 1xn
        P1[1] = 1.0 # only extract the second vertex 
        P1 = P1.T
        # TODO this could be cached, only required if rhits changes
        Ainv, Bn = computeDerivatives(rhits) # Ainv: dim: nxn, Bn: dim: nx1

        p1new = rhits[1].P() - beta * (Tp1 @ P1 @ Ainv @ Bn @ Tpn.T @ dp)
        p0dir = p1new - rhits[0].P()
        rhitsnew = [rhits[0]]

        # trace new hits
        ray2 = Ray(rhits[0].P(), p0dir)
        prevPlane = hits[-1].Plane() # plane at P
        cplane = rhits[-1].Plane() # plane at C0 (final plane)

        for j in range(1, len(rhits) - 1):
            hit2 = closestIntersect(ray2, prevPlane)
            if hit2 is None:
                break # TODO intersect with P-plane
            rhitsnew.append(hit2)
            ray2 = ray2.transfer(hit2)
            ray2 = ray2.sampleNext(hit2)
            if ray2 is None:
                break
            prevPlane = hit2.Plane()
        
        # final intersection with C0 plane
        if ray2 is not None:
            hit2 = ray2.calcHit(cplane, forceIntersect=True)
            if hit2.T() > 0:
                rhitsnew.append(hit2)

        foundBetter = False
        if len(rhitsnew) != len(rhits):
            print(f"ME {i}: expected {len(rhits)} hits, got {len(rhitsnew)} hits, reducing beta.")
        else:
            # check if error got smaller
            dpnew = C0 - rhitsnew[-1].P()
            if np.linalg.norm(dpnew) < np.linalg.norm(dp):
                rhits = rhitsnew
                print(f"ME {i}: improved solution with |dp|={np.linalg.norm(dpnew):.4g}, beta={beta:.4g}.")
                beta = min(1.0, beta * 2.0)
                foundBetter = True
        
        if not foundBetter:
            beta = beta * 0.5
            print(f"ME {i}: no improvement, reducing beta to {beta:.4g}.")

    newDir = rhits[1].P() - rhits[0].P()
    return newDir / np.linalg.norm(newDir)



# reverses hits so that they go from P to C0
def reverseHits(C0, dir, hits, includeP=False):
    virtualT = np.dot(C0 - hits[0].P(), -dir)
    virtualC1 = hits[0].P() - dir * virtualT # position of C1 on the virtual plane defined by C0 with normal dir

    rhits = []
    if includeP:
        P = hits[-1].P()
        dirn = -dir if len(hits) < 2 else hits[-2].P() - hits[-1].P()
        rhits.append(Hit(Plane(P, P + [dirn[1], -dirn[0]]), P, 0)) # virtual plane at P
    
    for i in range(len(hits) - 2, -1, -1):
        rhit = Hit(hits[i].Plane(), hits[i].P(), hits[i+1].T())
        rhits.append(rhit)
    rhits.append(Hit(Plane(virtualC1, virtualC1 + [dir[1], -dir[0]]), virtualC1, virtualT)) # virtual plane at C0
    return rhits

def methodReverseRayDiff(C0, C1, dir, hits):
    if len(hits) == 0:
        return dir
    
    Jpp = np.identity(2)
    Jpd = np.zeros((2, 2))
    Jdp = np.zeros((2, 2))
    Jdd = np.identity(2)
    
    # perform a transfer from the hit[0] to the virtual plane at C0
    virtualT = np.dot(C0 - hits[0].P(), -dir)
    virtualC1 = hits[0].P() - dir * virtualT # position of C1 on the virtual plane defined by C0 with normal dir
    Jpp, Jpd, Jdp, Jdd = transferRRDiff(Jpp, Jpd, Jdp, Jdd, -dir, dir, virtualT)

    R = -dir # outgoing ray direction
    for i in range(1, len(hits)):
        # handle surface interactin at j
        j = i - 1
        D = hits[j].P() - hits[i].P() # incomming ray direction
        t = np.linalg.norm(D)
        D = D / t # normalize
        N = hits[j].Plane().N()
        ShadingN = N
        M = np.zeros((2, 2))
        if Ray.use_normal_differential:
            M = hits[j].Plane().ShapeMatrix(hits[j].P())
            ShadingN = hits[j].ShadingN()
        
        if hits[j].Plane().Ior() == 1.0:
            # reflection
            Jpp, Jpd, Jdp, Jdd = reflectRRDiff(Jpp, Jpd, Jdp, Jdd, D, ShadingN, M)
        else:
            # refraction
            eta = hits[j].Plane().Ior()
            if np.dot(N, D) < 0:
                eta = 1.0 / eta
            else:
                M = -M # invert shape matrix if normal is flipped (this will flip the normal calculated by the shape matrix)
                ShadingN = -ShadingN
            Jpp, Jpd, Jdp, Jdd = refractRRDiff(Jpp, Jpd, Jdp, Jdd, D, R, ShadingN, eta, M)

        # transfer from i to j
        Jpp, Jpd, Jdp, Jdd = transferRRDiff(Jpp, Jpd, Jdp, Jdd, D, N, t)
        R = D # update outgoing direction for next iteration

    # final step: solve dD0 = Jpd^-1 * (C0 - virtualC1)
    dPn = C0 - virtualC1
    dD0 = solveForDD0(Jpd, dPn, R)
    dDn = mul(Jdd, dD0)
    #newDir = R + dD0 # R is the initial direction of the ray starting from P (hits[-1])
    newDir = dir - dDn # negate because we traced backwards
    print(f"Final Reverse Ray Diff: dD0={dD0}, dDn={dDn}, newDir={newDir}")

    # create reverse hits (for drawing and iterations)
    rhits = reverseHits(C0, dir, hits)

    if draw_guess:
        # draw the ray differential from P that goes to C0 with initial differential dD0
        rray = Ray(hits[-1].P(), R) # reverse ray
        rray = rray.setdD(dD0)

        # iterate through hits in reverse and skip the last hit (we start after P)
        for rhit in rhits:
            curP = rhit.P()
            ax.plot([rray.P()[0], curP[0]], [rray.P()[1], curP[1]], 'b-', label=LABEL_RAY2 if rhit == rhits[0] else None)
            # transfer ray
            prevdP = rray.P() + rray.dP()
            rray = rray.transfer(rhit)
            assert np.linalg.norm(rray.P() - rhit.P()) < 1e-5 # should be the same
            curdP = rray.P() + rray.dP()
            if draw_differentials and iterations == 0:
                ax.plot([prevdP[0], curdP[0]], [prevdP[1], curdP[1]], 'b--', label=LABEL_RAY2_DIFF if rhit == rhits[0] else None)

            rray = rray.sampleNext(rhit) # should be simply invertable in our case

    if iteration_strategy == 0:
        newDir = doVirtualIterations(C0, newDir, hits)
    if iteration_strategy == 1:
        newDir = doRealIterations(C0, dir, newDir, hits)
    if iteration_strategy == 2:
        newDir = doReverseRealIterations(C0, dir, newDir, hits, initialDir0=R + dD0)
    
    newDir /= np.linalg.norm(newDir)
    return newDir


def draw_direction(C0, dir):
    ax.arrow(C0[0], C0[1], dir[0] * 1.5, dir[1] * 1.5, head_width=0.2, color='orange', length_includes_head=True, label=LABEL_RAY2_PRED)

def draw_prediction(C0, dir, hits, color='orange', label=LABEL_RAY2_PRED):
    if(len(hits) == 0):
        draw_direction(C0, dir)
        return

    ray2 = Ray(C0, dir)
    # draw predicted ray2 path
    for hit in hits:
        hit2 = ray2.calcHit(hit.Plane(), forceIntersect=True)
        if hit2 is not None:
            ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], color, label=label if hit == hits[0] else None) # draw in orange
            
            prevP = ray2.P() + ray2.dP()
            ray2 = ray2.transfer(hit2)
            curP = ray2.P() + ray2.dP()
            if draw_differentials and iterations > 0:
                ax.plot([prevP[0], curP[0]], [prevP[1], curP[1]], color=color, linestyle='--', label=LABEL_RAY2_DIFF if hit == hits[0] else None)

            ray2 = ray2.sampleNext(hit2)
            if ray2 is None:
                if hit2.Plane().Ior() != 0.0:
                    # draw x to indicate that path ended due to refraction failure
                    ax.plot(hit2.P()[0], hit2.P()[1], 'rx', markersize=12)
                break # refraction not possible

def trace_and_draw_actual(C0, dir, hits, color='orange', label=LABEL_RAY2_PRED, prevPlane=None):
    if(len(hits) == 0):
        draw_direction(C0, dir)
        return
    ray2 = Ray(C0, dir)
    P = hits[-1].P()
    N = hits[-1].Plane().N()
    # face forward N
    if len(hits) >= 2:
        if np.dot(hits[-2].P() - hits[-1].P(), N) > 0:
            N = -N
    elif np.dot(dir, N) > 0:
        N = -N

    newHits = []
    bestDiff = float('inf')
    index = 0

    # draw actual ray2 path
    for _ in range(len(hits) + EXTRA_BOUNCES):
        hit = closestIntersect(ray2, prevPlane)
        if hit is None:
            break
        prevPlane = hit.Plane()
        newHits.append(hit)
        ray2 = ray2.transfer(hit)
        
        curP = ray2.P()
        diff = np.linalg.norm(P - curP)**2
        diff += max(0.0, np.dot(N, curP - P))**2 # penalize if in front of geometric plane (but not on or behind)

        if diff < bestDiff:
            bestDiff = diff
            index = len(newHits) - 1

        ray2 = ray2.sampleNext(hit)
        if ray2 is None:
            break # refraction not possible

    draw_prediction(C0, dir, newHits[:index+1], color, label)

# ---------------------------------------------------------------
# Reflect and Shear methods
# ---------------------------------------------------------------

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
        nextRay = ray.sampleNext(hit)
        if nextRay is not None:
            ray = nextRay
        cosbeta = abs(np.dot(ray.D(), hit.Plane().N()))
        speed *= cosalpha / cosbeta

    if draw_guess:
        end_point = C1 + dir * rayLength
        ax.plot([C0[0], end_point[0]], [C0[1], end_point[1]], 'b-', label=LABEL_RAY2)
        # print P* at end_point
        ax.plot(end_point[0], end_point[1], 'bo')
        ax.text(end_point[0]+0.2, end_point[1]+0.2, "P*", color='b')

    newDir = C1 + dir * rayLength - C0
    newDir /= np.linalg.norm(newDir)

    if iteration_strategy == 0:
        newDir = doVirtualIterations(C0, newDir, hits)
    if iteration_strategy == 1:
        newDir = doRealIterations(C0, dir, newDir, hits)
    #if iteration_strategy == 2:
    #    newDir = doReverseRealIterations(C0, dir, newDir, hits, initialDir0=-(ray2.D() + lastS * ray2.dD()))

    return newDir

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

    if draw_guess:
        ax.plot([C0[0], Pnew[0]], [C0[1], Pnew[1]], 'b-', label=LABEL_RAY2)
        # print P* at Pnew
        ax.plot(Pnew[0], Pnew[1], 'bo')
        ax.text(Pnew[0]+0.2, Pnew[1]+0.2, "P*", color='b')

    newDir = Pnew - C0
    newDir /= np.linalg.norm(newDir)

    if iteration_strategy == 0:
        newDir = doVirtualIterations(C0, newDir, hits)
    if iteration_strategy == 1:
        newDir = doRealIterations(C0, dir, newDir, hits)
    if iteration_strategy == 2:
        lastDir = -mul(np.linalg.inv(viewTransform[:2,:2]), newDir)
        newDir = doReverseRealIterations(C0, dir, newDir, hits, initialDir0=lastDir)

    return newDir

# ---------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.07, 0.1, 0.6, 0.8])  # main plot area (left)

# Slider panel
ax_sliders = [
    fig.add_axes([0.75, 0.95, 0.2, 0.03]),  # C1.x (moved up)
    fig.add_axes([0.75, 0.90, 0.2, 0.03]),  # C1.y
    fig.add_axes([0.75, 0.85, 0.2, 0.03]),  # C1 angle
    fig.add_axes([0.75, 0.80, 0.2, 0.03]),  # Draw Normals checkbox
    fig.add_axes([0.75, 0.75, 0.2, 0.03]),  # C0.x
    fig.add_axes([0.75, 0.70, 0.2, 0.03]),  # C0.y
    fig.add_axes([0.75, 0.65, 0.2, 0.03]),  # Ray.tangent_scale
    fig.add_axes([0.75, 0.60, 0.2, 0.03]),  # max bounces
    fig.add_axes([0.75, 0.55, 0.2, 0.03]),  # draw last iteration
    fig.add_axes([0.75, 0.50, 0.2, 0.03]),  # draw differentials
    fig.add_axes([0.75, 0.45, 0.2, 0.03]),  # draw guess
    fig.add_axes([0.75, 0.40, 0.2, 0.03]),  # iterations
    fig.add_axes([0.75, 0.28, 0.2, 0.1]),   # iteration strategy radio buttons
    fig.add_axes([0.75, 0.16, 0.2, 0.1]),   # predict strategy radio buttons
    fig.add_axes([0.75, 0.11, 0.2, 0.05]),   # use N differentials
    fig.add_axes([0.75, 0.06, 0.2, 0.05]),   # use speed
    fig.add_axes([0.75, 0.01, 0.2, 0.05]),   # use shear
]

slider_C1x = Slider(ax_sliders[0], "C1.x", -10.0, 10.0, valinit=C1[0])
slider_C1y = Slider(ax_sliders[1], "C1.y", -10.0, 10.0, valinit=C1[1])
slider_C1a = Slider(ax_sliders[2], "C1.angle", -180.0, 180.0, valinit=C1_angle)
checkbox_draw_normals = CheckButtons(ax_sliders[3], ["Draw Normals"], [draw_normals])
slider_C0x = Slider(ax_sliders[4], "C0.x", -10.0, 10.0, valinit=C0[0])
slider_C0y = Slider(ax_sliders[5], "C0.y", -10.0, 10.0, valinit=C0[1])
slider_tangent_scale = Slider(ax_sliders[6], "Ray.tangent_scale", 0.001, 0.5, valinit=Ray.tangent_scale)
# integer slider for max bounces
slider_max_bounces = Slider(ax_sliders[7], "Max Bounces", 1, 10, valinit=max_bounces, valstep=1)
# checkbox for draw last iteration
checkbox_draw_last_iteration = CheckButtons(ax_sliders[8], ["Draw Last Iteration"], [draw_last_iteration])
# checkbox for draw differentials and guess
checkbox_draw_differentials = CheckButtons(ax_sliders[9], ["Draw Differentials"], [draw_differentials])
checkbox_draw_guess = CheckButtons(ax_sliders[10], ["Draw Guess"], [draw_guess])
# integer slider for iterations
slider_iterations = Slider(ax_sliders[11], "Iterations ", 0, 20, valinit=iterations, valstep=1)
# radio buttons for iteration strategy
radio_iteration_strategy = RadioButtons(ax_sliders[12], iteration_strategies, active=iteration_strategy)
# radio buttons for predict strategy
radio_predict_strategy = RadioButtons(ax_sliders[13], predict_strategies, active=predict_strategy)
# checkbox for use N differentials
checkbox_use_n_differentials = CheckButtons(ax_sliders[14], ["Use N Diff."], [Ray.use_normal_differential])
# checkbox for use speed
checkbox_use_speed = CheckButtons(ax_sliders[15], ["Use Speed"], [useSpeed])
# checkbox for use shear
checkbox_use_shear = CheckButtons(ax_sliders[16], ["Use Shear"], [useShear])

# ---------------------------------------------------------------
# Slider callbacks
# ---------------------------------------------------------------

def update(val):
    global C0, C1, C1_angle, max_bounces, draw_differentials, draw_guess, draw_normals, iterations, iteration_strategy, predict_strategy, useSpeed, useShear, draw_last_iteration
    C1[0] = slider_C1x.val
    C1[1] = slider_C1y.val
    C1_angle = slider_C1a.val
    C0[0] = slider_C0x.val
    C0[1] = slider_C0y.val
    max_bounces = int(slider_max_bounces.val)
    draw_last_iteration = checkbox_draw_last_iteration.get_status()[0]
    draw_differentials = checkbox_draw_differentials.get_status()[0]
    draw_guess = checkbox_draw_guess.get_status()[0]
    draw_normals = checkbox_draw_normals.get_status()[0]
    iterations = int(slider_iterations.val)
    iteration_strategy = iteration_strategies.index(radio_iteration_strategy.value_selected)
    predict_strategy = predict_strategies.index(radio_predict_strategy.value_selected)
    Ray.tangent_scale = slider_tangent_scale.val
    Ray.use_normal_differential = checkbox_use_n_differentials.get_status()[0]
    useSpeed = checkbox_use_speed.get_status()[0]
    useShear = checkbox_use_shear.get_status()[0]
    draw_scene()

for s in [slider_C1x, slider_C1y, slider_C1a, slider_C0x, slider_C0y, slider_max_bounces, slider_tangent_scale, slider_iterations]:
    s.on_changed(update)

for c in [checkbox_draw_differentials, checkbox_draw_guess, checkbox_draw_normals, radio_iteration_strategy, radio_predict_strategy, checkbox_use_speed, checkbox_use_shear, checkbox_draw_last_iteration, checkbox_use_n_differentials]:
    c.on_clicked(update)

# Initial draw
draw_scene()
plt.show()
