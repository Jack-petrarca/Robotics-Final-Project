# ---------------------------------------------
# PUT CODE HERE TO CONTROL ROBOT MOVEMENT LOGIC
# ---------------------------------------------

# ---------- helpers ----------
def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

ranges = np.array(msg.ranges, dtype=np.float32)
ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)

N = len(ranges)
mid = N // 2

# sector sizes (robust for 360 or 720 scans)
front_w = max(5, N // 20)      # ~18 deg (360) / ~9 deg (720)
side_w  = max(8, N // 12)      # ~30 deg (360) / ~15 deg (720)

front = ranges[mid-front_w: mid+front_w]
left  = ranges[mid: mid+side_w]
right = ranges[mid-side_w: mid]

front_min = float(np.min(front))
left_avg  = float(np.mean(left))
right_avg = float(np.mean(right))

# ---------- lazy-init state (no __init__ edits) ----------
if not hasattr(self, "mode"):
    self.mode = "WANDER"
if not hasattr(self, "mode_until"):
    self.mode_until = 0.0
if not hasattr(self, "turn_dir"):
    self.turn_dir = 1.0  # +1 left, -1 right

t = float(self.elapsed)

# ---------- parameters ----------
omega_max = 1.2
k_heading = 1.8          # how strongly to steer to a desired heading

# obstacle thresholds (meters)
front_stop = 0.55        # too close -> hard avoid
front_caution = 0.85     # caution -> wall follow / gentle steer

# boundary box
edge_enter = 0.7         # enter boundary mode within this of ±10
edge_exit  = 1.5         # must come back this far inside to exit boundary mode
boundary_turn_gain = 1.6 # steer-to-center gain
boundary_omega_max = 1.2

near_edge = (self.x > 10-edge_enter) or (self.x < -10+edge_enter) or (self.y > 10-edge_enter) or (self.y < -10+edge_enter)
safe_inside = (self.x < 10-edge_exit) and (self.x > -10+edge_exit) and (self.y < 10-edge_exit) and (self.y > -10+edge_exit)

# ---------- mode transitions ----------
# Highest priority: boundary
if near_edge:
    self.mode = "BOUNDARY"

# Next: obstacle avoidance
elif front_min < front_stop:
    self.mode = "AVOID"
    # pick the more open side and remember it for a moment
    self.turn_dir = 1.0 if left_avg > right_avg else -1.0
    self.mode_until = t + 0.7  # commit time

elif self.mode == "AVOID" and t < self.mode_until:
    # stay in avoid mode until time expires
    pass

else:
    # default
    if self.mode != "WANDER":
        self.mode = "WANDER"

# ---------- compute omega by mode ----------
if self.mode == "BOUNDARY":
    # Steer toward the center (0,0) so we don't "orbit" the wall
    desired = math.atan2(-self.y, -self.x)
    err = wrap(desired - self.yaw)
    self.omega = boundary_turn_gain * err

    # clamp boundary omega
    if self.omega > boundary_omega_max:
        self.omega = boundary_omega_max
    elif self.omega < -boundary_omega_max:
        self.omega = -boundary_omega_max

    # exit boundary mode when safely inside again
    if safe_inside:
        self.mode = "WANDER"

elif self.mode == "AVOID":
    # hard turn away from obstacle, with a little forward "unstick" wobble
    self.omega = float(self.turn_dir) * 1.1 + 0.08 * math.sin(2.2 * t)

else:
    # WANDER: mix gentle right-wall-follow + slow bias changes so we explore
    # Right wall-follow: keep right side around d_des
    d_des = 0.9
    right_min = float(np.min(right))
    err = d_des - right_min
    omega_wall = 1.0 * err

    # If front is getting close, add steering away from the closer side
    omega_front = 0.0
    if front_min < front_caution:
        omega_front = 0.9 * (1.0 if left_avg > right_avg else -1.0)

    # Slow varying bias so we don't settle into a perfect circle
    omega_bias = 0.12 * math.sin(0.35 * t) + 0.06 * math.sin(0.11 * t)

    self.omega = omega_wall + omega_front + omega_bias

# final clamp + ensure float type for ROS message
if self.omega > omega_max:
    self.omega = omega_max
elif self.omega < -omega_max:
    self.omega = -omega_max

self.omega = float(self.omega)
