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
		
		self.starttime = 0.0
		self.elapsed = 0.0
		self.started = False

		# ---------- SPIRAL + AVOIDANCE (simple knobs) ----------
		self.v_cmd = 1.0                      # <-- speed knob (m/s). Increase to go faster.
		self.turn_spacing = 1.20              # meters between turns (120 cm)
		self.k_spiral = self.turn_spacing/(2.0*math.pi)  # Archimedean spiral r = k*theta
		self.lookahead = 0.7                  # radians ahead along spiral
		self.Kp = 2.6                         # heading gain
		self.wmax = 2.2                       # max turn rate (rad/s)

		# Avoidance measurements
		self.front_min = float('inf')
		self.front_bearing = 0.0              # radians (robot frame): 0 forward, +left, -right
		self.right_min = float('inf')

		# Avoidance behavior
		self.avoid = False
		self.des_right = 0.50                 # 10 cm closer than 0.60m
		self.obs_trigger = 1.00               # start avoiding if obstacle within 1.0 m
		self.path_tol = 0.35                  # rad: obstacle must be near desired heading to trigger
		# ------------------------------------------------------

		# ---------- FIX: unwrap theta so spiral keeps expanding ----------
		self.theta_prev = 0.0
		self.theta_unwrapped = 0.0
		# ---------------------------------------------------------------

		self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cbk, 10)
		self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cbk, 10)
		
		print('end node initialization')

	# ---------- helpers ----------
	def wrap_pi(self, a):
		while a > math.pi: a -= 2.0*math.pi
		while a < -math.pi: a += 2.0*math.pi
		return a

	def idx(self, ang, msg):
		i = int(round((ang - msg.angle_min)/msg.angle_increment))
		return max(0, min(len(msg.ranges)-1, i))
	# ----------------------------
		
	def odom_cbk(self, msg):
		self.x = msg.pose.pose.position.x
		self.y = msg.pose.pose.position.y
		self.yaw = math.atan2(
			2.0*(msg.pose.pose.orientation.w * msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y),
			1.0-2.0*(msg.pose.pose.orientation.y * msg.pose.pose.orientation.y + msg.pose.pose.orientation.z * msg.pose.pose.orientation.z)
		)
		
		if self.started == False:
			self.started = True
			self.starttime = msg.header.stamp.sec

			# init unwrap state
			theta0 = math.atan2(self.y, self.x)
			self.theta_prev = theta0
			self.theta_unwrapped = theta0

		self.elapsed = msg.header.stamp.sec - self.starttime
		print("elapsed", self.elapsed)

		# ---------- SPIRAL desired heading (NO stopping / NO r_limit) ----------
		theta_now = math.atan2(self.y, self.x)      # [-pi, pi]
		d = self.wrap_pi(theta_now - self.theta_prev)
		self.theta_unwrapped += d
		self.theta_prev = theta_now

		theta_use = max(0.0, self.theta_unwrapped)  # keep non-negative for r=k*theta
		theta_t = theta_use + self.lookahead
		r_t = self.k_spiral * theta_t

		x_t = r_t*math.cos(theta_t)
		y_t = r_t*math.sin(theta_t)

		hdg_des = math.atan2(y_t - self.y, x_t - self.x)
		hdg_err = self.wrap_pi(hdg_des - self.yaw)

		w_spiral = max(-self.wmax, min(self.wmax, self.Kp*hdg_err))
		# ----------------------------------------------------------------------

		# ---------- Avoid ONLY if obstacle is in desired direction ----------
		in_path = (self.front_min < self.obs_trigger) and (abs(self.wrap_pi(self.front_bearing - hdg_err)) < self.path_tol)
		if in_path:
			self.avoid = True
		elif self.front_min > (self.obs_trigger + 0.2):
			self.avoid = False
		# ------------------------------------------------------------------

		cmd = Twist()
		cmd.linear.x = self.v_cmd

		if not self.avoid:
			cmd.angular.z = w_spiral
		else:
			# Rounded right go-around: right-clearance + slight right bias
			w_follow = 1.2*(self.des_right - self.right_min) if math.isfinite(self.right_min) else 0.0
			w_avoid = w_follow - 0.35
			if self.front_min < 0.75:
				w_avoid -= 0.8
			cmd.angular.z = max(-self.wmax, min(self.wmax, w_avoid))

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

		# ---------- front + right measurements for avoidance ----------
		ranges = np.array(msg.ranges, dtype=float)

		# front sector [-30deg,+30deg]
		i0 = self.idx(-math.radians(30), msg)
		i1 = self.idx(+math.radians(30), msg)
		if i0 > i1: i0, i1 = i1, i0
		fs = ranges[i0:i1+1]
		fm = np.isfinite(fs) & (fs > msg.range_min) & (fs < msg.range_max)
		if np.any(fm):
			k = int(np.where(fm)[0][np.argmin(fs[fm])])
			self.front_min = float(fs[fm].min())
			self.front_bearing = msg.angle_min + msg.angle_increment*(i0 + k)
		else:
			self.front_min = float('inf')
			self.front_bearing = 0.0

		# right sector [-120deg,-60deg]
		r0 = self.idx(-math.radians(120), msg)
		r1 = self.idx(-math.radians(60), msg)
		if r0 > r1: r0, r1 = r1, r0
		rs = ranges[r0:r1+1]
		rm = np.isfinite(rs) & (rs > msg.range_min) & (rs < msg.range_max)
		self.right_min = float(rs[rm].min()) if np.any(rm) else float('inf')
		# -----------------------------------------------------------

		self.map.show_map()
			

def main(args=None):
	rclpy.init(args=args)
	
	node = Project()
	
	rclpy.spin(node)
	
	node.destroy_node()
	rclpy.shutdown()
	
if __name__ == '__main__':
	main()
