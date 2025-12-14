import rclpy
import math
import numpy as np
import cv2

from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class MyMap():
	def __init__(self, xmin, xmax, ymin, ymax, res):
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.res = res
		self.W = math.ceil((xmax - xmin)/res)
		self.H = math.ceil((ymax - ymin)/res)
		self.map = np.full((self.H, self.W), 100, dtype=np.uint8)

	def __del__(self):
		cv2.imwrite('my_map.png', self.map)

	def world2map(self, x, y):
		v = int(np.floor((x - self.xmin)/self.res))
		u = int(np.floor((self.ymax - y)/self.res))
		return v, u

	# PATCH: full ray traversal restored
	def insert_ray(self, vm, um, v, u, occupied):
		free = 255

		dv = abs(v - vm)
		du = abs(u - um)
		i, j = vm, um
		step_i = 1 if v > vm else -1
		step_j = 1 if u > um else -1

		if dv > du:
			err = dv // 2
			while i != v:
				if 0 < j < self.H and 0 < i < self.W:
					self.map[j, i] = free
				err -= du
				if err < 0:
					j += step_j
					err += dv
				i += step_i
		else:
			err = du // 2
			while j != u:
				if 0 < j < self.H and 0 < i < self.W:
					self.map[j, i] = free
				err -= dv
				if err < 0:
					i += step_i
					err += du
				j += step_j

		if occupied and 0 < u < self.H and 0 < v < self.W:
			self.map[u, v] = 0

	def show_map(self):
		cv2.imshow("map", self.map)
		cv2.waitKey(10)


class Project(Node):
	def __init__(self):
		super().__init__('myfirstosnode')

		self.map = MyMap(-10, 10, -10, 10, 0.05)

		self.x = 0.0
		self.y = 0.0
		self.yaw = 0.0

		self.known_pillars = []
		self.visited = []
		self.target_index = 0
		self.have_target = False
		self.tx = 0.0
		self.ty = 0.0

		# PATCH: escape timer
		self.escape_timer = 0.0

		self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.create_subscription(Odometry, '/odom', self.odom_cbk, 10)
		self.create_subscription(LaserScan, '/scan', self.scan_cbk, 10)

	def odom_cbk(self, msg):
		self.x = msg.pose.pose.position.x
		self.y = msg.pose.pose.position.y
		q = msg.pose.pose.orientation
		self.yaw = math.atan2(
			2*(q.w*q.z + q.x*q.y),
			1 - 2*(q.y*q.y + q.z*q.z)
		)

	def detect_pillars(self, msg):
		pillars = []
		cluster = []
		threshold = 0.15

		for i, r in enumerate(msg.ranges):
			if r < msg.range_max:
				if not cluster:
					cluster = [(i, r)]
				elif abs(r - cluster[-1][1]) < threshold:
					cluster.append((i, r))
				else:
					if len(cluster) >= 2:
						pillars.append(cluster)
					cluster = [(i, r)]
			else:
				if len(cluster) >= 2:
					pillars.append(cluster)
				cluster = []

		return pillars

	def cluster_to_world(self, cluster, msg):
		i_mean = int(np.mean([i for i, r in cluster]))
		r_mean = np.mean([r for i, r in cluster])
		theta = msg.angle_min + i_mean * msg.angle_increment
		xr = r_mean * math.cos(theta)
		yr = r_mean * math.sin(theta)
		xw = self.x + xr * math.cos(self.yaw) - yr * math.sin(self.yaw)
		yw = self.y + xr * math.sin(self.yaw) + yr * math.cos(self.yaw)
		return xw, yw

	def scan_cbk(self, msg):
		vm, um = self.map.world2map(self.x, self.y)

		for i, r in enumerate(msg.ranges):
			theta = msg.angle_min + i * msg.angle_increment
			rr = min(r, msg.range_max)
			x = self.x + rr * math.cos(theta + self.yaw)
			y = self.y + rr * math.sin(theta + self.yaw)
			v, u = self.map.world2map(x, y)
			self.map.insert_ray(vm, um, v, u, r < msg.range_max)

		pillars = self.detect_pillars(msg)
		for cluster in pillars:
			xw, yw = self.cluster_to_world(cluster, msg)
			if all(math.hypot(xw-px, yw-py) > 0.8 for px, py in self.known_pillars):
				self.known_pillars.append((xw, yw))
				print(f"Detected pillar at {xw:.2f}, {yw:.2f}")

		if not self.have_target and self.target_index < len(self.known_pillars):
			self.tx, self.ty = self.known_pillars[self.target_index]
			self.have_target = True

		self.map.show_map()

		cmd = Twist()

		# PATCH: escape behavior
		if self.escape_timer > 0.0:
			cmd.linear.x = -0.1
			cmd.angular.z = 0.6
			self.escape_timer -= 0.1
			self.cmd_pub.publish(cmd)
			return

		if self.have_target:
			dx = self.tx - self.x
			dy = self.ty - self.y
			dist = math.hypot(dx, dy)
			err = math.atan2(
				math.sin(math.atan2(dy, dx) - self.yaw),
				math.cos(math.atan2(dy, dx) - self.yaw)
			)

			if dist < 0.55:
				self.visited.append((self.tx, self.ty))
				print(f"VISITED pillar at {self.tx:.2f}, {self.ty:.2f}")
				self.target_index += 1
				self.have_target = False
				self.escape_timer = 1.0    # PATCH
				return

			cmd.linear.x = 0.3
			cmd.angular.z = 1.5 * err
		else:
			cmd.angular.z = 0.6

		self.cmd_pub.publish(cmd)


def main(args=None):
	rclpy.init(args=args)
	node = Project()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
