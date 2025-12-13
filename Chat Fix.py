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
		print('init map')
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
		
	def saveMap(self):
		cv2.imwrite('my_map.png', self.map)
		
	def map2world(self, v, u):
		x = self.res*(v+0.5) + self.xmin
		y = self.res*(u+0.5) + self.ymax
		return x, y
		
	def world2map(self, x, y):
		v = np.floor((x - self.xmin)/self.res).astype(int)
		u = np.floor((self.ymax - y)/self.res).astype(int)
		return v, u
		
	def insert_obsticle(self, v, u):
		self.map[u, v] = 0
		
	def insert_ray(self, vm, um, v, u, occupied):
		free = 255
		dv = abs(v-vm)
		du = abs(u-um)
		
		i = vm
		j = um
		
		step_i = 1 if v > vm else -1
		step_j = 1 if u > um else -1
		
		if dv > du:
			err = dv // 2
			while i != v:
				if not (i == vm and j == um):
					if 0 < j < self.W and 0 < i < self.H:
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
					if 0 < j < self.W and 0 < i < self.H:
						self.map[j, i] = free
						
				err -= dv
				if err < 0:
					i += step_i
					err += du
				j += step_j
		
		if occupied:
			if 0 < v < self.W and 0 < u < self.H:
				self.map[u, v] = 0
		else:
			if 0 < v < self.W and 0 < u < self.H:
				self.map[u, v] = free
					
	def show_map(self):
		cv2.imshow("map", self.map)
		cv2.waitKey(10)


class Project(Node):
	def __init__(self):
		print('starting node initialization...')
		super().__init__('myfirstosnode')
		
		self.map = MyMap(-10.0, 10.0, -10.0, 10.0, 0.05)
		self.timer_period = 0.1
		self.b = 0.1
		self.x = 0.0
		self.y = 0.0
		self.yaw = 0.0
		self.omega = 0.0
		
		# --- Pillar targeting state ---
		self.visited = []
		self.visit_radius = 0.7  # Back to reasonable value
		self.have_target = False
		self.tx = 0.0
		self.ty = 0.0
		
		# --- Escape state to prevent getting stuck ---
		self.escaping = False
		self.escape_counter = 0
		self.ESCAPE_DURATION = 15  # Shorter escape
		self.escape_angle = 0.0  # Direction to escape towards
		
		self.starttime = 0.0
		self.elapsed = 0.0
		self.started = False
		
		self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cbk, 10)
		self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cbk, 10)
		
		print('end node initialization')
		
	def odom_cbk(self, msg):
		self.x = msg.pose.pose.position.x
		self.y = msg.pose.pose.position.y
		self.yaw = math.atan2(
			2.0*(msg.pose.pose.orientation.w * msg.pose.pose.orientation.z +
			     msg.pose.pose.orientation.x * msg.pose.pose.orientation.y),
			1.0-2.0*(msg.pose.pose.orientation.y**2 +
			          msg.pose.pose.orientation.z**2)
		)
		
		if not self.started:
			self.started = True
			self.starttime = msg.header.stamp.sec
			
		self.elapsed = msg.header.stamp.sec - self.starttime
		print("elapsed", self.elapsed)
		
	def detect_pillars(self, msg):
		pillars = []
		cluster = []
		threshold = 0.15
		
		for i, r in enumerate(msg.ranges):
			if r < msg.range_max:
				if not cluster:
					cluster = [(i, r)]
				else:
					if abs(r - cluster[-1][1]) < threshold:
						cluster.append((i, r))
					else:
						if len(cluster) > 5:
							pillars.append(cluster)
						cluster = [(i, r)]
			else:
				if len(cluster) > 5:
					pillars.append(cluster)
				cluster = []
		
		return pillars

	def cluster_to_world(self, cluster, msg):
		indices = [i for i, r in cluster]
		ranges = [r for i, r in cluster]
		
		i_mean = int(np.mean(indices))
		r_mean = np.mean(ranges)
		
		theta = msg.angle_min + i_mean * msg.angle_increment
		
		xr = r_mean * math.cos(theta)
		yr = r_mean * math.sin(theta)
		
		xw = self.x + xr * math.cos(self.yaw) - yr * math.sin(self.yaw)
		yw = self.y + xr * math.sin(self.yaw) + yr * math.cos(self.yaw)
		
		return xw, yw

	def is_visited(self, x, y):
		for vx, vy in self.visited:
			if math.hypot(x - vx, y - vy) < self.visit_radius:
				return True
		return False
		
	def scan_cbk(self, msg):
		beta = msg.angle_increment
		i = 0
		
		pRW = np.array([[self.x], [self.y]])
		RRW = np.array([[math.cos(self.yaw), -math.sin(self.yaw)],
		                [math.sin(self.yaw),  math.cos(self.yaw)]])
						
		vm, um = self.map.world2map(self.x, self.y)
						
		for r in msg.ranges:
			theta_i = msg.angle_min + beta*i
			if r < msg.range_max:
				DR = np.array([[r*math.cos(theta_i)],
				               [r*math.sin(theta_i)]])
				occupied = True
			else:
				DR = np.array([[msg.range_max*math.cos(theta_i)],
				               [msg.range_max*math.sin(theta_i)]])
				occupied = False
			
			DW = RRW @ DR + pRW
			v, u = self.map.world2map(DW[0], DW[1])
			self.map.insert_ray(vm, um, v, u, occupied)
			i += 1
		
		self.map.show_map()
		
		cmd = Twist()
		
		# Handle escape mode first
		if self.escaping:
			self.escape_counter += 1
			if self.escape_counter < self.ESCAPE_DURATION:
				# Move away from the pillar in the escape direction
				cmd.linear.x = 0.3  # Drive forward away from pillar
				
				# Turn towards escape angle
				angle_error = math.atan2(
					math.sin(self.escape_angle - self.yaw),
					math.cos(self.escape_angle - self.yaw)
				)
				cmd.angular.z = 2.0 * angle_error
				
				self.cmd_pub.publish(cmd)
				return
			else:
				# Done escaping
				self.escaping = False
				self.escape_counter = 0
		
		# Normal pillar detection and targeting
		pillars = self.detect_pillars(msg)
		
		targets = []
		for cluster in pillars:
			xw, yw = self.cluster_to_world(cluster, msg)
			if not self.is_visited(xw, yw):
				dist = math.hypot(xw - self.x, yw - self.y)
				targets.append((dist, xw, yw))
		
		# Debug: print visited and detected pillars
		if len(targets) > 0 or len(pillars) > len(targets):
			print(f"Detected {len(pillars)} pillars, {len(targets)} unvisited, {len(self.visited)} total visited")
		
		if targets:
			targets.sort()
			_, self.tx, self.ty = targets[0]
			self.have_target = True
		else:
			self.have_target = False
		
		if self.have_target:
			dx = self.tx - self.x
			dy = self.ty - self.y
			dist = math.hypot(dx, dy)
			
			target_angle = math.atan2(dy, dx)
			angle_error = math.atan2(
				math.sin(target_angle - self.yaw),
				math.cos(target_angle - self.yaw)
			)
			
			COLLECT_RADIUS = 0.50
			
			if dist > COLLECT_RADIUS:
				# Approaching the pillar
				cmd.angular.z = 1.5 * angle_error
				cmd.linear.x = 0.3
			else:
				# Within collection radius - mark as visited FIRST, then escape
				self.visited.append((self.tx, self.ty))
				print(f"Visited pillar at ({self.tx:.2f}, {self.ty:.2f}), total: {len(self.visited)}")
				self.have_target = False
				self.escaping = True
				self.escape_counter = 0
				# Start the escape maneuver immediately
				cmd.linear.x = -0.25
				cmd.angular.z = 0.6
		
		elif self.searching:
			# Active search mode - spin in place to look for pillars
			self.search_counter += 1
			if self.search_counter < self.SEARCH_DURATION:
				print(f"Searching for pillars... {self.search_counter}/{self.SEARCH_DURATION}")
				cmd.linear.x = 0.0
				cmd.angular.z = 0.8 * self.search_turn_direction
			else:
				# Done searching, switch to exploration mode
				print("Search complete, switching to exploration")
				self.searching = False
				self.search_counter = 0
				# Alternate turn direction for next search
				self.search_turn_direction *= -1

		else:
			# Exploration mode - move forward while avoiding walls
			left_min = min(msg.ranges[0:90])
			right_min = min(msg.ranges[270:360])
			front_min = min(msg.ranges[165:195])  # Check front
			
			# If obstacle ahead, turn more aggressively
			if front_min < 0.5:
				cmd.linear.x = 0.1
				cmd.angular.z = 1.0 if left_min > right_min else -1.0
			else:
				cmd.linear.x = 0.4
				cmd.angular.z = 0.25 * (left_min - right_min)
		
		self.cmd_pub.publish(cmd)


def main(args=None):
	rclpy.init(args=args)
	node = Project()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
