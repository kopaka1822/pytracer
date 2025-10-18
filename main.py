import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from plane import Plane

# ---------------------------------------------------------------
# Scene setup: finite line segments
# ---------------------------------------------------------------

planes = [
    Plane([-10, -5], [10, -5]),       # bottom
    Plane([-8, -3], [-3, -1]),        # slanted left
    Plane([3, -1], [8, -3]),          # slanted right
    Plane([-5, 5], [-2, 8]),          # upper left
    Plane([2, 7], [5, 5]),            # upper right
    Plane([-1, 8], [1, 8])            # top
]

# Camera positions (modifiable via sliders)
C0 = np.array([-8.6, 5.0])
C1 = np.array([-9.2, 4.5])
C1_angle = -46.8  # in degrees

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
    ax.plot(C0[0], C0[1], 'bo', label="C0")
    ax.plot(C1[0], C1[1], 'go', label="C1")

    # Draw C1 ray direction
    dir_len = 1.5
    dir = np.array([np.cos(np.radians(C1_angle)), np.sin(np.radians(C1_angle))])
    ax.arrow(C1[0], C1[1], dir[0]*dir_len, dir[1]*dir_len,
             head_width=0.2, color='g', length_includes_head=True)

    fig.canvas.draw_idle()

# ---------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.07, 0.1, 0.6, 0.8])  # main plot area (left)

# Slider panel
ax_sliders = [
    fig.add_axes([0.75, 0.75, 0.2, 0.03]),  # C1.x
    fig.add_axes([0.75, 0.70, 0.2, 0.03]),  # C1.y
    fig.add_axes([0.75, 0.65, 0.2, 0.03]),  # C1 angle
    fig.add_axes([0.75, 0.55, 0.2, 0.03]),  # C0.x
    fig.add_axes([0.75, 0.50, 0.2, 0.03])   # C0.y
]

slider_C1x = Slider(ax_sliders[0], "C1.x", -10.0, 10.0, valinit=C1[0])
slider_C1y = Slider(ax_sliders[1], "C1.y", -10.0, 10.0, valinit=C1[1])
slider_C1a = Slider(ax_sliders[2], "C1.angle", -180.0, 180.0, valinit=C1_angle)
slider_C0x = Slider(ax_sliders[3], "C0.x", -10.0, 10.0, valinit=C0[0])
slider_C0y = Slider(ax_sliders[4], "C0.y", -10.0, 10.0, valinit=C0[1])

# ---------------------------------------------------------------
# Slider callbacks
# ---------------------------------------------------------------

def update(val):
    global C0, C1, C1_angle
    C1[0] = slider_C1x.val
    C1[1] = slider_C1y.val
    C1_angle = slider_C1a.val
    C0[0] = slider_C0x.val
    C0[1] = slider_C0y.val
    draw_scene()

for s in [slider_C1x, slider_C1y, slider_C1a, slider_C0x, slider_C0y]:
    s.on_changed(update)

# Initial draw
draw_scene()
plt.show()
