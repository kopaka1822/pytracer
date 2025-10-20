import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider, RadioButtons
from plane import Plane
from ray import Ray
from hit import Hit

# ---------------------------------------------------------------
# Scene setup: finite line segments
# ---------------------------------------------------------------

reflection_scene = [
    Plane([-10, -5], [10, -5]),       # bottom
    Plane([-8, -3], [-3, -1]),        # slanted left
    Plane([3, -1], [8, -3]),          # slanted right
    Plane([-5, 5], [-2, 8]),          # upper left
    Plane([2, 7], [5, 5]),            # upper right
    Plane([-1, 8], [1, 8]),           # top


    # box around -10, 10
    Plane([-10, -10], [-10, 10]),
    Plane([-10, 10], [10, 10]),
    Plane([10, 10], [10, -10]),
    Plane([10, -10], [-10, -10]),
]

refraction_scene = [

    # two planes in center for glass
    Plane([-10, 2], [10, 2], ior=1.5),
    Plane([10, 0], [-10, 0], ior=1.5),

    # box around -10, 10
    Plane([-10, -10], [-10, 10]),
    Plane([-10, 10], [10, 10]),
    Plane([10, 10], [10, -10]),
    Plane([10, -10], [-10, -10]),
]

planes = refraction_scene

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

# solves A * s = B for scalar s
def solveLinearEq(A: np.ndarray, B: np.ndarray) -> float:
    if abs(A[0]) > abs(A[1]):
        s = B[0] / A[0]
    else:
        s = B[1] / A[1]
    return s


# Camera positions (modifiable via sliders)
C0 = np.array([-8.6, 5.0])
C1 = np.array([-9.2, 4.5])
C1_angle = -46.8  # in degrees
max_bounces = 3
draw_differentials = True
draw_guess = True
guess_strategies = ["same direction", "initial intersection"]
guess_strategy = 0  # index into guess_strategies
predict_strategies = ["adjust dD", "adjust dP", "adjust D"]
predict_strategy = 0  # index into predict_strategies

# ---------------------------------------------------------------
# Draw function
# ---------------------------------------------------------------

def draw_scene():
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_title("Ray Differential Motion Vector")
    ax.grid(True, linestyle="--", alpha=0.3)

    LABEL_RAY = "original"
    LABEL_RAY2 = "guess"
    LABEL_RAY2_DIFF = "differential"
    LABEL_RAY2_PRED = "predicted"

    # Draw finite planes
    for pl in planes:
        P1, P2 = pl.P1(), pl.P2()
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 'k-', lw=1.5)
        # draw normal at midpoint
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
    ray = Ray(C1, dir)
    initial_ray2 = Ray(C0, dir)
    ray2 = initial_ray2
    lastS = 0.0 # last solution for ray2 differential
    hits = []
    # main loop
    for i in range(max_bounces):
        # Find the closest intersection
        hit = closestIntersect(ray, prevPlane)
        if hit is None:
            break
        hits.append(hit)

        if i == 0 and guess_strategy == 1: # initial intersection
            initial_ray2 = Ray(C0, hit.P() - C0)
            ray2 = initial_ray2


        # Draw the ray to the hit point
        ax.plot([ray.P()[0], hit.P()[0]], [ray.P()[1], hit.P()[1]], 'g-', label=LABEL_RAY if i == 0 else None)

        # Transfer the ray to the hit point
        ray = ray.transfer(hit)
        ray = ray.sampleNext(hit)
        prevPlane = hit.Plane()

        # intersect ray2 with the same plane
        hit2 = ray2.calcHit(hit.Plane(), forceIntersect=True)
        if hit2 is not None:
            if draw_guess:
                ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'b-', label=LABEL_RAY2 if i == 0 else None)
            # update ray2
            prevP = ray2.P() + ray2.dP()
            ray2 = ray2.transfer(hit2)
            curP = ray2.P() + ray2.dP()
            # draw ray differential segment
            if draw_differentials:
                ax.plot([prevP[0], curP[0]], [prevP[1], curP[1]], 'c--', label=LABEL_RAY2_DIFF if i == 0 else None)
            ray2 = ray2.sampleNext(hit2)

            # calc current solution for ray2 differential (PStar + s * dP = P <=> s * dP = P - PStar)
            P = ray.P()
            PStar = ray2.P()
            dP = ray2.dP()
            s = solveLinearEq(dP, P - PStar)
            lastS = s

    # use lastS to determine the new ray2 initial direction.
    ray2 = initial_ray2
    newDir = ray2.D() + lastS * ray2.dD() # adjust dD (predict_strategy == 0)
    # determine new direction based on transferred dP
    if predict_strategy == 1:
        ray2TempTransfer = ray2.transfer(ray2.calcHit(hits[0].Plane(), forceIntersect=True))
        newDir = ray2TempTransfer.P() + lastS * ray2TempTransfer.dP() - C0
    if predict_strategy == 2:
        ray2TempShift = initial_ray2.shiftS(lastS)
        newDir = ray2TempShift.D()
    newDir /= np.linalg.norm(newDir)
    ray2 = Ray(C0, newDir)
    # draw predicted ray2 path
    for hit in hits:
        hit2 = ray2.calcHit(hit.Plane(), forceIntersect=True)
        if hit2 is not None:
            ax.plot([ray2.P()[0], hit2.P()[0]], [ray2.P()[1], hit2.P()[1]], 'orange', label=LABEL_RAY2_PRED if hit == hits[0] else None) # draw in orange
            ray2 = ray2.transfer(hit2)
            ray2 = ray2.sampleNext(hit2)

    ax.legend(loc="upper right")
    fig.canvas.draw_idle()

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
    fig.add_axes([0.75, 0.75, 0.2, 0.03]),  # C0.x
    fig.add_axes([0.75, 0.70, 0.2, 0.03]),  # C0.y
    fig.add_axes([0.75, 0.65, 0.2, 0.03]),  # Ray.tangent_scale
    fig.add_axes([0.75, 0.60, 0.2, 0.03]),  # max bounces
    fig.add_axes([0.75, 0.50, 0.2, 0.03]),  # draw differentials
    fig.add_axes([0.75, 0.45, 0.2, 0.03]),  # draw guess
    fig.add_axes([0.75, 0.30, 0.2, 0.1]),   # guess strategy radio buttons
    fig.add_axes([0.75, 0.20, 0.2, 0.1]),   # predict strategy radio buttons
]

slider_C1x = Slider(ax_sliders[0], "C1.x", -10.0, 10.0, valinit=C1[0])
slider_C1y = Slider(ax_sliders[1], "C1.y", -10.0, 10.0, valinit=C1[1])
slider_C1a = Slider(ax_sliders[2], "C1.angle", -180.0, 180.0, valinit=C1_angle)
slider_C0x = Slider(ax_sliders[3], "C0.x", -10.0, 10.0, valinit=C0[0])
slider_C0y = Slider(ax_sliders[4], "C0.y", -10.0, 10.0, valinit=C0[1])
slider_tangent_scale = Slider(ax_sliders[5], "Ray.tangent_scale", 0.001, 0.5, valinit=Ray.tangent_scale)
# integer slider for max bounces
slider_max_bounces = Slider(ax_sliders[6], "Max Bounces", 1, 10, valinit=max_bounces, valstep=1)
# checkbox for draw differentials and guess
checkbox_draw_differentials = CheckButtons(ax_sliders[7], ["Draw Differentials"], [draw_differentials])
checkbox_draw_guess = CheckButtons(ax_sliders[8], ["Draw Guess"], [draw_guess])
# radio buttons for guess strategy
radio_guess_strategy = RadioButtons(ax_sliders[9], guess_strategies, active=guess_strategy)
radio_predict_strategy = RadioButtons(ax_sliders[10], predict_strategies, active=predict_strategy)

# ---------------------------------------------------------------
# Slider callbacks
# ---------------------------------------------------------------

def update(val):
    global C0, C1, C1_angle, max_bounces, draw_differentials, draw_guess, guess_strategy, predict_strategy
    C1[0] = slider_C1x.val
    C1[1] = slider_C1y.val
    C1_angle = slider_C1a.val
    C0[0] = slider_C0x.val
    C0[1] = slider_C0y.val
    max_bounces = int(slider_max_bounces.val)
    draw_differentials = checkbox_draw_differentials.get_status()[0]
    draw_guess = checkbox_draw_guess.get_status()[0]
    guess_strategy = guess_strategies.index(radio_guess_strategy.value_selected)
    predict_strategy = predict_strategies.index(radio_predict_strategy.value_selected)
    Ray.tangent_scale = slider_tangent_scale.val
    draw_scene()

for s in [slider_C1x, slider_C1y, slider_C1a, slider_C0x, slider_C0y, slider_max_bounces, slider_tangent_scale]:
    s.on_changed(update)

for c in [checkbox_draw_differentials, checkbox_draw_guess, radio_guess_strategy, radio_predict_strategy]:
    c.on_clicked(update)

# Initial draw
draw_scene()
plt.show()
