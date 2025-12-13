# --- Go toward nearest pillar in front (starter) ---

ranges = np.array(msg.ranges, dtype=np.float32)
ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)

N = len(ranges)
mid = N // 2

# look +/- ~30 degrees around the front
w = max(10, N // 12)

front_ranges = ranges[mid-w : mid+w]
k_min = int(np.argmin(front_ranges))
r_min = float(front_ranges[k_min])

# bearing of that minimum (relative to robot forward), in radians
i_min = (mid - w) + k_min
theta = msg.angle_min + msg.angle_increment * i_min

# turn toward it: omega = k * bearing
k_turn = 1.8
self.omega = float(k_turn * theta)

# if nothing close (all range_max-ish), just slowly spin to search
if r_min > 2.5:               # tune this
    self.omega = 0.4

# clamp so Twist gets reasonable values
omega_max = 1.2
if self.omega > omega_max:
    self.omega = omega_max
elif self.omega < -omega_max:
    self.omega = -omega_max

self.omega = float(self.omega)
