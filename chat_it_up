import rclpy
import math
import numpy as np
import cv2

from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


# ---------- Map class (fixed bounds + cleaner math) ----------
class MyMap:
    def __init__(self, xmin, xmax, ymin, ymax, res):
        print("init map")
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.res = float(res)

        self.W = int(math.ceil((self.xmax - self.xmin) / self.res))  # cols (v)
        self.H = int(math.ceil((self.ymax - self.ymin) / self.res))  # rows (u)

        # Start unknown-ish gray
        self.map = np.full((self.H, self.W), 100, dtype=np.uint8)

    def __del__(self):
        # Save on shutdown if possible
        try:
            cv2.imwrite("my_map.png", self.map)
        except Exception:
            pass

    def saveMap(self, filename="my_map.png"):
        cv2.imwrite(filename, self.map)

    def map2world(self, v, u):
        # v = col index (x), u = row index (y)
        x = self.res * (v + 0.5) + self.xmin
        y = self.ymax - self.res * (u + 0.5)
        return x, y

    def world2map(self, x, y):
        v = int(np.floor((x - self.xmin) / self.res))
        u = int(np.floor((self.ymax - y) / self.res))
        return v, u

    def in_bounds(self, v, u):
        return (0 <= v < self.W) and (0 <= u < self.H)

    def insert_obstacle(self, v, u):
        if self.in_bounds(v, u):
            self.map[u, v] = 0  # occupied

    def insert_ray(self, vm, um, v, u, occupied):
        """
        Draw a ray from (vm,um) to (v,u) marking free along the path and
        setting the end cell as occupied/free depending on 'occupied'.
        Uses Bresenham-style stepping.
        """
        free = 255

        # If start or end are wildly out of bounds, we still step but only write if in bounds
        dv = abs(v - vm)
        du = abs(u - um)

        i = vm  # col
        j = um  # row

        step_i = 1 if v > vm else -1
        step_j = 1 if u > um else -1

        if dv > du:
            err = dv // 2
            while i != v:
                if not (i == vm and j == um):
                    if self.in_bounds(i, j):
                        self.map[j, i] = free

                err -= du
                if err < 0:
                    j += step_j
                    err += dv
                i += step_i
        else:
            err = du // 2
            while j != u:
                if not (j == um and i == vm):
                    if self.in_bounds(i, j):
                        self.map[j, i] = free

                err -= dv
                if err < 0:
                    i += step_i
                    err += du
                j += step_j

        # Mark the endpoint
        if self.in_bounds(v, u):
            self.map[u, v] = 0 if occupied else free

    def show_map(self):
        cv2.imshow("map", self.map)
        cv2.waitKey(10)


# ---------- Main ROS2 Node ----------
class Project(Node):
    def __init__(self):
        print("starting node initialization...")
        super().__init__("myfirstrosnode")

        # Map
        self.map = MyMap(-10.0, 10.0, -10.0, 10.0, 0.05)

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Time bookkeeping
        self.starttime = 0.0
        self.elapsed = 0.0
        self.started = False
        self.saved = False

        # “Touch many pillars” controller state
        self.state = "SEEK"         # SEEK -> APPROACH -> LEAVE
        self.close_dist = 0.35      # consider pillar “visited” inside this distance
        self.leave_time = 1.0       # seconds to back off + turn
        self.leave_start = 0.0

        self.visits = 0
        self.last_visit_time = -999.0

        # Publishers/subscribers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cbk, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cbk, 10)

        print("end node initialization")

    def odom_cbk(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # yaw from quaternion (planar)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        # Timer based on message header stamp (seconds)
        if not self.started:
            self.started = True
            self.starttime = float(msg.header.stamp.sec)

        self.elapsed = float(msg.header.stamp.sec) - self.starttime
        # print("elapsed", self.elapsed)

        # Stop at 90s and save map once
        if (self.elapsed >= 90.0) and (not self.saved):
            self.map.saveMap("mymap.png")
            self.saved = True
            self.get_logger().info(f"Saved map at t={self.elapsed:.1f}s, visits={self.visits}")

        # Optional hard stop after 90s:
        if self.elapsed >= 90.0:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

    def scan_cbk(self, msg):
        # If we haven’t started timing yet, don’t do anything fancy
        if not self.started:
            return

        # Convert scan to numpy
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment

        # Valid returns
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)

        # Focus on front half so we don’t chase stuff behind us
        front = (angles > -np.pi / 2) & (angles < np.pi / 2)
        valid_front = valid & front

        # Nearest “pillar” candidate
        best_r = None
        best_a = None
        if np.any(valid_front):
            inds = np.where(valid_front)[0]
            best_i = inds[np.argmin(ranges[inds])]
            best_r = float(ranges[best_i])
            best_a = float(angles[best_i])

        # --- Mapping (occupancy update) ---
        pRW = np.array([[self.x], [self.y]], dtype=np.float32)
        RRW = np.array(
            [[math.cos(self.yaw), -math.sin(self.yaw)],
             [math.sin(self.yaw),  math.cos(self.yaw)]],
            dtype=np.float32
        )

        vm, um = self.map.world2map(self.x, self.y)

        beta = msg.angle_increment
        for i, r in enumerate(ranges):
            theta_i = msg.angle_min + beta * i

            if np.isfinite(r) and (r > msg.range_min) and (r < msg.range_max):
                # occupied endpoint
                DR = np.array([[r * math.cos(theta_i)], [r * math.sin(theta_i)]], dtype=np.float32)
                DW = RRW @ DR + pRW
                v, u = self.map.world2map(float(DW[0]), float(DW[1]))
                self.map.insert_ray(vm, um, v, u, True)
            else:
                # free ray to max range
                rmax = msg.range_max
                DR = np.array([[rmax * math.cos(theta_i)], [rmax * math.sin(theta_i)]], dtype=np.float32)
                DW = RRW @ DR + pRW
                v, u = self.map.world2map(float(DW[0]), float(DW[1]))
                self.map.insert_ray(vm, um, v, u, False)

        # --- Safety: if too close straight ahead, escape ---
        cmd = Twist()
        front_cone = (angles > -0.35) & (angles < 0.35)
        front_valid = np.isfinite(ranges) & front_cone
        front_min = float(np.min(ranges[front_valid])) if np.any(front_valid) else msg.range_max

        if front_min < 0.22:
            cmd.linear.x = 0.0
            cmd.angular.z = 1.2
            self.cmd_pub.publish(cmd)
            self.map.show_map()
            return

        # --- Pillar-seeking state machine ---
        t = float(self.elapsed)

        if self.state == "SEEK":
            if best_r is None:
                # roam
                cmd.linear.x = 0.25
                cmd.angular.z = 0.4
            else:
                self.state = "APPROACH"

        elif self.state == "APPROACH":
            if best_r is None:
                self.state = "SEEK"
            else:
                err = best_a  # bearing error (rad), already robot frame
                cmd.angular.z = 1.8 * err

                # go faster when aligned
                cmd.linear.x = 0.6 * max(0.0, (1.0 - abs(err)))

                # “visited” condition with debounce so 1 pillar doesn't spam counts
                if (best_r < self.close_dist) and ((t - self.last_visit_time) > 1.0):
                    self.visits += 1
                    self.last_visit_time = t
                    self.get_logger().info(f"VISIT #{self.visits} at t={t:.1f}s (r={best_r:.2f}m)")
                    self.state = "LEAVE"
                    self.leave_start = t

        elif self.state == "LEAVE":
            # back off + turn so we break contact and can hit more pillars
            cmd.linear.x = -0.15
            cmd.angular.z = 1.0
            if (t - self.leave_start) > self.leave_time:
                self.state = "SEEK"

        # Clamp angular velocity
        cmd.angular.z = float(np.clip(cmd.angular.z, -1.5, 1.5))

        # If we’re past 90 seconds, don’t move
        if self.elapsed >= 90.0:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)
        self.map.show_map()


def main(args=None):
    rclpy.init(args=args)

    node = Project()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
