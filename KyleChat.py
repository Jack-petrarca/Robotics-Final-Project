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

    # --- Spiral search params ---
		self.spiral_enabled = True

		self.v_spiral = 0.35          # m/s
		self.omega0 = 1.0             # rad/s (tight start)
		self.omega_min = 0.12         # rad/s (wide later)
		self.omega_decay = 0.03       # rad/s per second

		self.omega_avoid = 0.0        # scan-based correction

		
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

		if self.spiral_enabled:
			v, omega_spiral = self.spiral_motion()
			cmd.linear.x = v
			cmd.angular.z = omega_spiral + self.omega_avoid
			cmd.angular.z = max(-1.5, min(1.5, cmd.angular.z))
		else:
			cmd.linear.x = 0.5
			cmd.angular.z = self.omega_avoid

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
			
			
#	def control_callback(self):
#		self.x
#		self.y
#		self.yaw
#		ex = self.xd - self.x
#		ey = self.yd - self.y
		
#		u1 = self.kx * ex
#		u2 = self.ky * ey
			
#		v = math.cos(self.yaw)*u1 + math.sin(self.yaw)*u2
		
#		omega = -math.sin(self.yaw)*u1/self.b + math.cos(self.yaw)*u2/self.b
		
#		twist_msg = Twist()
#		twist_msg.linear.x = v
#		twist_msg.angular.z = omega
#		self.cmd_publisher.publish(twist_msg)
		
#		print(self.x, self.y, self.yaw)
		


def main(args=None):
	rclpy.init(args=args)
	
	node = Project()
	
	rclpy.spin(node)
	
	node.destroy_node()
	rclpy.shutdown()
	
if __name__ == '__main__':
	main() 
