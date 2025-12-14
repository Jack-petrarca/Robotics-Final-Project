import rclpy
import math
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Pillar goals (world frame)
        self.known_pillars = []
        self.visited_pillars = []
        self.current_goal_index = 0
        self.have_goal = False

        self.goal_x = 0.0
        self.goal_y = 0.0

        # --- NEW: emergency collision info ---
        self.front_min = float('inf')

        # ROS interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # Control loop
        self.timer = self.create_timer(0.2, self.control_callback)

    # --------------------------------------------------
    # Odometry callback
    # --------------------------------------------------
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    # --------------------------------------------------
    # Laser scan + pillar detection
    # --------------------------------------------------
    def scan_callback(self, msg):
        # --- NEW: compute front_min for emergency avoidance ---
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)

        N = len(ranges)
        mid = N // 2
        w_front = max(10, N // 16)  # ~20-25 degrees cone
        self.front_min = float(np.min(ranges[mid - w_front: mid + w_front]))

        # --- existing clustering-based detection (with a tiny robustness tweak) ---
        clusters = []
        cluster = []
        threshold = 0.15

        for i, r in enumerate(msg.ranges):
            # ignore invalids and max range (treat as gap)
            if (not math.isfinite(r)) or (r <= msg.range_min) or (r >= msg.range_max):
                if len(cluster) >= 2:
                    clusters.append(cluster)
                cluster = []
                continue

            # build clusters based on neighbor similarity
            if not cluster:
                cluster = [(i, r)]
            elif abs(r - cluster[-1][1]) < threshold:
                cluster.append((i, r))
            else:
                if len(cluster) >= 2:
                    clusters.append(cluster)
                cluster = [(i, r)]

        if len(cluster) >= 2:
            clusters.append(cluster)

        for c in clusters:
            i_mean = int(np.mean([i for i, r in c]))
            r_mean = float(np.mean([r for i, r in c]))
            theta = msg.angle_min + i_mean * msg.angle_increment

            xr = r_mean * math.cos(theta)
            yr = r_mean * math.sin(theta)

            xw = self.x + xr * math.cos(self.yaw) - yr * math.sin(self.yaw)
            yw = self.y + xr * math.sin(self.yaw) + yr * math.cos(self.yaw)

            # larger association tolerance
            already_known = False
            for px, py in self.known_pillars:
                if math.hypot(xw - px, yw - py) < 0.8:
                    already_known = True
                    break

            if not already_known:
                self.known_pillars.append((xw, yw))
                self.get_logger().info(f"Detected pillar at x={xw:.2f}, y={yw:.2f}")

    # --------------------------------------------------
    # Control loop (go-to-goal with standoff + collision override)
    # --------------------------------------------------
    def control_callback(self):
        msg = Twist()

        # --- NEW: emergency collision override (highest priority) ---
        EMERGENCY_STOP = 0.35  # tune: bigger = safer (0.30-0.45 typical)
        if self.front_min < EMERGENCY_STOP:
            msg.linear.x = 0.0
            msg.angular.z = 0.9  # spin to clear
            self.cmd_vel_pub.publish(msg)
            return

        # --- Assign next pillar goal (standoff point) ---
        if (not self.have_goal) and (self.current_goal_index < len(self.known_pillars)):
            px, py = self.known_pillars[self.current_goal_index]

            # vector from pillar to robot
            vx = self.x - px
            vy = self.y - py
            d = math.hypot(vx, vy)

            # NEW: standoff distance so we don't ram the pillar
            STANDOFF = 0.60  # meters (close enough to count, far enough not to hit)
            if d > 1e-3:
                self.goal_x = px + (vx / d) * STANDOFF
                self.goal_y = py + (vy / d) * STANDOFF
            else:
                self.goal_x, self.goal_y = px, py

            self.have_goal = True
            self.get_logger().info(f"New goal (standoff): x={self.goal_x:.2f}, y={self.goal_y:.2f}")

        # No goal → rotate to scan environment
        if not self.have_goal:
            msg.linear.x = 0.0
            msg.angular.z = 0.6
            self.cmd_vel_pub.publish(msg)
            return

        # Go-to-goal controller
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        distance = math.hypot(dx, dy)

        angle_to_goal = math.atan2(dy, dx)
        angle_error = math.atan2(
            math.sin(angle_to_goal - self.yaw),
            math.cos(angle_to_goal - self.yaw)
        )

        # Stop when close to standoff goal (NOT the pillar itself)
        STOP_DISTANCE = 0.20   # close to the standoff point
        TURN_CUTOFF = 0.75

        if distance < STOP_DISTANCE:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

            # mark visited (still record the actual detected pillar point)
            px, py = self.known_pillars[self.current_goal_index]
            self.visited_pillars.append((px, py))
            self.get_logger().info(f"VISITED pillar at x={px:.2f}, y={py:.2f}")

            self.current_goal_index += 1
            self.have_goal = False

        else:
            K_lin = 0.6
            K_ang = 1.6

            linear_speed = min(K_lin * distance, 0.4)
            angular_speed = max(min(K_ang * angle_error, 1.0), -1.0)

            # slow down near goal to avoid overshoot
            if distance < TURN_CUTOFF:
                linear_speed *= 0.5

            msg.linear.x = float(linear_speed)
            msg.angular.z = float(angular_speed)

        self.cmd_vel_pub.publish(msg)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
