import rclpy
import math
import numpy as np
import cv2

from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


#Added from 12/9 Lecture
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
		#self.map[v, u] = 0
		
	def insert_ray(self, vm, um, v, u, occupied):
		free = 255
		dv = abs(v-vm)
		du = abs(u-um)
		
		i = vm
		j = um
		
		step_i = 1 if v > vm else - 1
		step_j = 1 if u > um else -1
		
		if dv > du:
			err = dv // 2
			while i != v:
				if not (i == vm and j == um):
					if j > 0 and j < self.W:
						if i > 0 and i < self.H:
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
					if j > 0 and j < self.W:
						if i > 0 and i < self.H:
							self.map[j, i] = free
						
				err -= dv
				if err < 0:
					i += step_i
					err += du
				j += step_j
		
				
		if occupied == True:
			if v > 0 and v < self.W:
				if u > 0 and u < self.H:
					self.map [u, v] = 0
					
		else:
			if v > 0 and v < self.W:
				if u > 0 and u < self.H:
					self.map [u, v] = free
					
		
					
	def show_map(self):
		cv2.imshow("map", self.map)
		cv2.waitKey(10)
#end add from 12/9
		

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

		
#############################
		# --- Spiral search params ---
		self.spiral_enabled = True

		self.v_spiral = 0.35          # m/s
		self.omega0 = 1.0             # rad/s (tight start)
		self.omega_min = 0.12         # rad/s (wide later)
		self.omega_decay = 0.03       # rad/s per second

		self.omega_avoid = 0.0        # scan-based correction
	# --- Search / pillar behavior ---
		self.mode = "SPIRAL"   # SPIRAL, APPROACH, ESCAPE
		self.target_angle = 0.0
		self.target_dist = None

		self.pillar_dist_thresh = 0.8     # detect pillar within 80 cm
		self.approach_dist = 0.45          # stop ~45 cm away
		self.safe_dist = 0.30              # hard safety stop

		self.last_pillar_time = 0.0
############################

		
		self.starttime = 0.0
		self.elapsed = 0.0
		self.started = False
		
		self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cbk, 10)
		self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cbk, 10)
		#self.control_timer = self.create_timer(self.timer_period, self.control_callback)
		
		print('end node initialization')

	def spiral_motion(self):
		t = float(self.elapsed)
		omega_mag = max(self.omega_min, self.omega0 - self.omega_decay * t)
		omega_spiral = +omega_mag   # CCW; flip sign for CW
		v = self.v_spiral
		return v, omega_spiral

	def detect_pillar(self, msg):
		ranges = np.array(msg.ranges)

		# Ignore invalid readings
		ranges[ranges < msg.range_min] = np.inf
		ranges[ranges > msg.range_max] = np.inf

		# Look only forward +/- 60 degrees
		center = len(ranges) // 2
		window = int(math.radians(60) / msg.angle_increment)

		sub = ranges[center-window:center+window]

		idx = np.argmin(sub)
		r_min = sub[idx]

		# Pillar signature: local min surrounded by farther points
		if r_min < self.pillar_dist_thresh:
			left = sub[max(idx-5, 0)]
			right = sub[min(idx+5, len(sub)-1)]

			if left - r_min > 0.15 and right - r_min > 0.15:
				angle = (idx - window) * msg.angle_increment
				return True, angle, r_min

		return False, None, None



		
	def odom_cbk(self, msg):
		self.x = msg.pose.pose.position.x
		self.y = msg.pose.pose.position.y
		self.yaw = math.atan2(2.0*(msg.pose.pose.orientation.w * msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y), 1.0-2.0*(msg.pose.pose.orientation.y * msg.pose.pose.orientation.y + msg.pose.pose.orientation.z * msg.pose.pose.orientation.z));
		
		if self.started == False:
			self.started = True
			self.starttime = msg.header.stamp.sec
		self.elapsed = msg.header.stamp.sec - self.starttime
		print("elapsed", self.elapsed)
		
		cmd = Twist()

		# ---------------- SPIRAL ----------------
		if self.mode == "SPIRAL":
			v, omega_spiral = self.spiral_motion()
			cmd.linear.x = v
			cmd.angular.z = omega_spiral + self.omega_avoid

		# ---------------- APPROACH ----------------
		elif self.mode == "APPROACH":
			cmd.angular.z = 1.5 * self.target_angle
			cmd.linear.x = 0.2

			if self.target_dist is not None and self.target_dist < self.approach_dist:
				self.mode = "ESCAPE"

			if self.target_dist is not None and self.target_dist < self.safe_dist:
				cmd.linear.x = 0.0

		# ---------------- ESCAPE ----------------
		elif self.mode == "ESCAPE":
			cmd.linear.x = 0.2
			cmd.angular.z = 1.0

			if self.elapsed - self.last_pillar_time > 3.0:
				self.mode = "SPIRAL"

		self.cmd_pub.publish(cmd)



		
		
	def scan_cbk(self, msg):
		beta = msg.angle_increment
		i = 0
		pRW = np.array([[self.x],
						[self.y]])
		RRW = np.array([[math.cos(self.yaw),-math.sin(self.yaw)],
						[math.sin(self.yaw),math.cos(self.yaw)]])
						
		vm, um, = self.map.world2map(self.x, self.y)
						
		for r in msg.ranges:
			if r<msg.range_max:
				theta_i = msg.angle_min + beta*i
				DR = np.array([[r*math.cos(theta_i)],
								[r*math.sin(theta_i)]])
				DW = RRW @	DR +pRW
				v, u = self.map.world2map(DW[0], DW[1])
				self.map.insert_ray(vm, um, v, u, True)
			else:
				theta_i = msg.angle_min + beta*i
				DR = np.array([[msg.range_max*math.cos(theta_i)],
								[msg.range_max*math.sin(theta_i)]])
				DW = RRW @	DR +pRW
				v, u = self.map.world2map(DW[0], DW[1])
				self.map.insert_ray(vm, um, v, u, False)
			i = i+1
		
		if self.elapsed == 10:
			cv2.imwrite("mymap.png", self.map.map)
			
			
		
		left_min = min(msg.ranges[0:90])
		right_min = min(msg.ranges[270:360])
		self.omega_avoid = 0.2*(left_min - right_min)
    	
		self.map.show_map()

		found, angle, dist = self.detect_pillar(msg)

		
		if self.mode == "APPROACH" and found:
			self.target_angle = angle
			self.target_dist = dist
		

		if self.mode == "SPIRAL" and found:
			self.mode = "APPROACH"
			self.target_angle = angle
			self.target_dist = dist
			self.last_pillar_time = self.elapsed



def main(args=None):
	rclpy.init(args=args)
	
	node = Project()
	
	rclpy.spin(node)
	
	node.destroy_node()
	rclpy.shutdown()
	
if __name__ == '__main__':
	main() 
