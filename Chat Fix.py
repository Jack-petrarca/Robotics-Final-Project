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
	
	def mark_visited_pillar(self, x, y, radius=0.6):
		"""Mark a visited pillar on the map with value 50 (gray)"""
		v, u = self.world2map(x, y)
		radius_cells = int(radius / self.res)
		for i in range(-radius_cells, radius_cells + 1):
			for j in range(-radius_cells, radius_cells + 1):
				if i*i + j*j <= radius_cells*radius_cells:
					vi = v + i
					uj = u + j
					if 0 <= vi < self.W and 0 <= uj < self.H:
						self.map[uj, vi] = 50  # Mark as visited (gray)
					
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
		self.visit_radius = 0.55  # Detection radius for visited pillars
		self.have_target = False
		self.tx = 0.0
		self.ty = 0.0
		self.stuck_counter = 0  # Detect when stuck
		self.last_position = (0.0, 0.0)
		
		# --- Exploration when no target ---
		self.explore_direction = 0.0
		self.explore_timer = 0
		
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
		
	def detect_pillars(self, msg):
		"""Detect pillar clusters from LIDAR data"""
		pillars = []
		cluster = []
		threshold = 0.15  # Max distance between consecutive points in a cluster
		
		for i, r in enumerate(msg.ranges):
			if r < msg.range_max:
				if not cluster:
					cluster = [(i, r)]
				else:
					# Check distance continuity
					if abs(r - cluster[-1][1]) < threshold:
						cluster.append((i, r))
					else:
						# End current cluster
						if len(cluster) > 5:  # Minimum cluster size
							pillars.append(cluster)
						cluster = [(i, r)]
			else:
				# End of valid range
				if len(cluster) > 5:
					pillars.append(cluster)
				cluster = []
		
		# Don't forget last cluster
		if len(cluster) > 5:
			pillars.append(cluster)
		
		return pillars

	def cluster_to_world(self, cluster, msg):
		"""Convert cluster to world coordinates"""
		indices = [i for i, r in cluster]
		ranges = [r for i, r in cluster]
		
		i_mean = int(np.mean(indices))
		r_mean = np.mean(ranges)
		
		theta = msg.angle_min + i_mean * msg.angle_increment
		
		# Robot frame coordinates
		xr = r_mean * math.cos(theta)
		yr = r_mean * math.sin(theta)
		
		# World frame coordinates
		xw = self.x + xr * math.cos(self.yaw) - yr * math.sin(self.yaw)
		yw = self.y + xr * math.sin(self.yaw) + yr * math.cos(self.yaw)
		
		return xw, yw

	def is_visited(self, x, y):
		"""Check if a pillar location has been visited"""
		for vx, vy in self.visited:
			if math.hypot(x - vx, y - vy) < self.visit_radius:
				return True
		return False
	
	def get_min_front_distance(self, msg):
		"""Get minimum distance in front of robot"""
		# Check front 60 degrees
		front_indices = list(range(150, 210))
		front_ranges = [msg.ranges[i] for i in front_indices if i < len(msg.ranges)]
		return min(front_ranges) if front_ranges else msg.range_max
	
	def is_stuck(self):
		"""Detect if robot is stuck (not moving)"""
		dist_moved = math.hypot(self.x - self.last_position[0], self.y - self.last_position[1])
		
		if dist_moved < 0.05:  # Moved less than 5cm
			self.stuck_counter += 1
		else:
			self.stuck_counter = 0
		
		self.last_position = (self.x, self.y)
		
		return self.stuck_counter > 20  # Stuck for 2 seconds (20 * 0.1s)
		
	def scan_cbk(self, msg):
		beta = msg.angle_increment
		i = 0
		
		pRW = np.array([[self.x], [self.y]])
		RRW = np.array([[math.cos(self.yaw), -math.sin(self.yaw)],
		                [math.sin(self.yaw),  math.cos(self.yaw)]])
						
		vm, um = self.map.world2map(self.x, self.y)
						
		# Update map with LIDAR data
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
		
		# Get obstacle info for safety
		min_front_dist = self.get_min_front_distance(msg)
		
		# Check if stuck
		if self.is_stuck():
			print(f"⚠ Robot stuck! Attempting recovery...")
			# Recovery: turn and abandon current target
			cmd.linear.x = 0.0
			cmd.angular.z = 1.0
			self.have_target = False
			self.stuck_counter = 0
			self.cmd_pub.publish(cmd)
			return
		
		# Detect pillars
		pillars = self.detect_pillars(msg)
		
		# Find unvisited targets
		targets = []
		for cluster in pillars:
			xw, yw = self.cluster_to_world(cluster, msg)
			if not self.is_visited(xw, yw):
				dist = math.hypot(xw - self.x, yw - self.y)
				targets.append((dist, xw, yw))
		
		# Update target (choose closest unvisited pillar)
		if targets:
			targets.sort()
			_, self.tx, self.ty = targets[0]
			self.have_target = True
		else:
			self.have_target = False
		
		# === MAIN CONTROL LOGIC ===
		if self.have_target:
			# Navigate to target pillar
			dx = self.tx - self.x
			dy = self.ty - self.y
			dist = math.hypot(dx, dy)
			
			target_angle = math.atan2(dy, dx)
			angle_error = math.atan2(
				math.sin(target_angle - self.yaw),
				math.cos(target_angle - self.yaw)
			)
			
			# Collection parameters
			COLLECTION_RADIUS = 0.50  # Success distance
			STOP_THRESHOLD = 0.35     # Emergency stop distance
			
			# Emergency stop if obstacle very close
			if min_front_dist < STOP_THRESHOLD:
				cmd.linear.x = 0.0
				cmd.angular.z = 0.0
				
				# If close to target, collect it
				if dist < 0.55 and not self.is_visited(self.tx, self.ty):
					self.visited.append((self.tx, self.ty))
					self.map.mark_visited_pillar(self.tx, self.ty, self.visit_radius)
					print(f"✓ COLLECTED #{len(self.visited)} at ({self.tx:.2f}, {self.ty:.2f}) [EMERGENCY STOP] - Time: {self.elapsed:.1f}s")
					self.have_target = False
				else:
					# Give up on this target if we can't reach it
					if self.stuck_counter > 15:
						print(f"✗ Abandoning unreachable target at ({self.tx:.2f}, {self.ty:.2f})")
						self.have_target = False
						self.stuck_counter = 0
			
			# Reached collection distance
			elif dist <= COLLECTION_RADIUS:
				cmd.linear.x = 0.0
				cmd.angular.z = 0.0
				
				if not self.is_visited(self.tx, self.ty):
					self.visited.append((self.tx, self.ty))
					self.map.mark_visited_pillar(self.tx, self.ty, self.visit_radius)
					print(f"✓ COLLECTED #{len(self.visited)} at ({self.tx:.2f}, {self.ty:.2f}) - Time: {self.elapsed:.1f}s")
				
				self.have_target = False
				self.explore_timer = 0
			
			# Final approach - very careful
			elif dist <= 0.70:
				# Only move if well-aligned
				if abs(angle_error) < 0.25 and min_front_dist > 0.40:
					cmd.linear.x = 0.12  # Slow final approach
					cmd.angular.z = 1.2 * angle_error
				else:
					# Rotate to align
					cmd.linear.x = 0.0
					cmd.angular.z = 1.5 * angle_error
			
			# Medium distance
			elif dist <= 1.5:
				if abs(angle_error) > 0.4:
					# Turn more
					cmd.linear.x = 0.15
					cmd.angular.z = 1.8 * angle_error
				else:
					# Move forward with caution
					if min_front_dist > 0.7:
						cmd.linear.x = 0.25
					else:
						cmd.linear.x = 0.15
					cmd.angular.z = 1.3 * angle_error
			
			# Far distance
			else:
				if abs(angle_error) > 0.5:
					cmd.linear.x = 0.18
					cmd.angular.z = 2.0 * angle_error
				else:
					# Speed based on clearance
					if min_front_dist > 2.0:
						cmd.linear.x = 0.35
					elif min_front_dist > 1.0:
						cmd.linear.x = 0.28
					else:
						cmd.linear.x = 0.20
					cmd.angular.z = 1.4 * angle_error
		
		else:
			# No visible unvisited pillars - explore
			self.explore_timer += 1
			
			# Get obstacle information
			left_min = min(msg.ranges[45:135]) if len(msg.ranges) > 135 else msg.range_max
			right_min = min(msg.ranges[225:315]) if len(msg.ranges) > 315 else msg.range_max
			
			# Obstacle avoidance
			if min_front_dist < 0.5:
				# Turn away from obstacle
				if left_min > right_min:
					cmd.angular.z = 1.0  # Turn left
				else:
					cmd.angular.z = -1.0  # Turn right
				cmd.linear.x = 0.0
				self.explore_timer = 0
			
			elif min_front_dist < 1.0:
				# Pick new explore direction
				if left_min > right_min:
					self.explore_direction = self.yaw + math.pi / 3
				else:
					self.explore_direction = self.yaw - math.pi / 3
				
				angle_error = math.atan2(
					math.sin(self.explore_direction - self.yaw),
					math.cos(self.explore_direction - self.yaw)
				)
				cmd.linear.x = 0.18
				cmd.angular.z = 1.2 * angle_error
				self.explore_timer = 0
			
			elif self.explore_timer > 40:
				# Change direction periodically
				self.explore_direction = self.yaw + math.pi / 2.5
				self.explore_timer = 0
			
			else:
				# Normal exploration
				if self.explore_timer == 0:
					self.explore_direction = self.yaw + math.pi / 6
				
				angle_error = math.atan2(
					math.sin(self.explore_direction - self.yaw),
					math.cos(self.explore_direction - self.yaw)
				)
				
				# Exploration speed
				if min_front_dist > 2.0:
					cmd.linear.x = 0.35
				elif min_front_dist > 1.5:
					cmd.linear.x = 0.28
				else:
					cmd.linear.x = 0.22
				
				cmd.angular.z = 1.0 * angle_error
		
		self.cmd_pub.publish(cmd)


def main(args=None):
	rclpy.init(args=args)
	node = Project()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
