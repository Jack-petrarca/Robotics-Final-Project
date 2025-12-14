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
    # Laser scan + pillar detection (PATCHED)
    # --------------------------------------------------
    def scan_callback(self, msg):
        clusters = []
        cluster = []
        threshold = 0.15

        for i, r in enumerate(msg.ranges):
            if r < msg.range_max:
                if not cluster:
                    cluster = [(i, r)]
                elif abs(r - cluster[-1][1]) < threshold:
                    cluster.append((i, r))
                else:
                    if len(cluster) >= 2:              # PATCH 1
                        clusters.append(cluster)
                    cluster = [(i, r)]
            else:
                if len(cluster) >= 2:                  # PATCH 1
                    clusters.append(cluster)
                cluster = []

        for c in clusters:
            i_mean = int(np.mean([i for i, r in c]))
            r_mean = np.mean([r for i, r in c])
            theta = msg.angle_min + i_mean * msg.angle_increment

            xr = r_mean * math.cos(theta)
            yr = r_mean * math.sin(theta)

            xw = self.x + xr * math.cos(self.yaw) - yr * math.sin(self.yaw)
            yw = self.y + xr * math.sin(self.yaw) + yr * math.cos(self.yaw)

            # PATCH 2: larger association tolerance
            already_known = False
            for px, py in self.known_pillars:
                if math.hypot(xw - px, yw - py) < 0.8:
                    already_known = True
                    break

            if not already_known:
                self.known_pillars.append((xw, yw))
                self.get_logger().info(
                    f"Detected pillar at x={xw:.2f}, y={yw:.2f}"
                )

    # --------------------------------------------------
    # Control loop (go-to-goal with safe stand-off)
    # --------------------------------------------------
    def control_callback(self):
        msg = Twist()

        # Assign next pillar (no sorting)
        if not self.have_goal and self.current_goal_index < len(self.known_pillars):
            self.goal_x, self.goal_y = self.known_pillars[self.current_goal_index]
            self.have_goal = True
            self.get_logger().info(
                f"New goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}"
            )

        # No goal → rotate to scan environment
        if not self.have_goal:
            msg.linear.x = 0.0
            msg.angular.z = 0.6
            self.cmd_vel_pub.publish(msg)
            return

        # Go-to-goal controller
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        angle_to_goal = math.atan2(dy, dx)
        angle_error = math.atan2(
            math.sin(angle_to_goal - self.yaw),
            math.cos(angle_to_goal - self.yaw)
        )

        STOP_DISTANCE = 0.55
        TURN_CUTOFF = 0.75

        if distance < STOP_DISTANCE:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

            self.visited_pillars.append((self.goal_x, self.goal_y))
            self.get_logger().info(
                f"VISITED pillar at x={self.goal_x:.2f}, y={self.goal_y:.2f}"
            )

            self.current_goal_index += 1
            self.have_goal = False

        else:
            K_lin = 0.5
            K_ang = 1.5

            linear_speed = min(K_lin * distance, 0.4)
            angular_speed = max(min(K_ang * angle_error, 1.0), -1.0)

            if distance < TURN_CUTOFF:
                linear_speed *= 0.5
                angular_speed *= 0.3

            msg.linear.x = linear_speed
            msg.angular.z = angular_speed

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
