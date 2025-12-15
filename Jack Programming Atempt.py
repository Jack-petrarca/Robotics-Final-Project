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
	#	self.timer_period = 0.1
	#	self.b = 0.1
		self.x = 0.0
		self.y = 0.0
		self.yaw = 0.0 
		self.omega = 0.0

		self.known_pillars = []    #Creates an empty array where pillars will be added once found
		self.visited = []          #keeps track of known pillars so we dont revisit them
		self.target_index = 0      #indexes through the known pillar array
		self.have_target = False   
		self.tx = 0.0              #target x and y in world frame
		self.ty = 0.0

		self.escape_timer = 0.0   #helps to clear current pillar before moving on
		
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
		self.yaw = math.atan2(2.0*(msg.pose.pose.orientation.w * msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y), 1.0-2.0*(msg.pose.pose.orientation.y * msg.pose.pose.orientation.y + msg.pose.pose.orientation.z * msg.pose.pose.orientation.z));
		
		if self.started == False:
			self.started = True
			self.starttime = msg.header.stamp.sec
		self.elapsed = msg.header.stamp.sec - self.starttime
		print("elapsed", self.elapsed)

	def detect_pillars(self,msg): #THIS FUNCTION ONLY PROVIDES WHAT THE ROBOT THINKS COULD BE AN OBJECT
		pillars = [] #stores detected pillars
		cluster = [] #This stores groups of Lidar scans to detemine if an object is present
		threshold = 0.15 #this determines the maximum distance the lidar scans must be apart from each other to be considered two different objects

		for i, r in enumerate(msg.ranges): #i is the index of the laser ray and r is the distance measurement
			if r < msg.range_max: # if laser maxed out range then it saw nothing and the entire length of the ray is free
				if not cluster: #if we are not building a cluster start a cluster (cluster of rays)
					cluster = [(i,r)]
				elif abs(r - cluster[-1][1]) < threshold: # compare current ray to previous ray's distance  if the distance is small both rays most likly hit the same object (threshold defines the min distance between rays for it to be considered the same object)
					cluster.append((i,r)) # add to cluster if both rays hit the same object
				else:
					if len(cluster) >= 2: #defines min about of rays in a cluster to be considered an object in this case its 2 or more rays
						pillars.append(cluster)
					cluster = [(i,r)]
			else:
				if len(cluster) >= 2: #an object jsut ended save if it is large enough and reset the cluster count.
					pillars.append(cluster)
				cluster = []
		return pillars

	def cluster_to_world(self, cluster, msg): #computes pillars from robot frame to world frame to provide a stable x y for the robot to target
		i_mean = int(np.mean([i for i, r in cluster])) #takes the average ray index to aproximate center of the pillars
		r_mean = np.mean([r for i, r in cluster]) #takes the average ray distance to aproximate center of the pillars
		theta = msg.angle_min + i_mean * msg.angle_increment #converts ray index to degrees in radians since the laser scanner outputs angle min and angle increments
		xr = r_mean * math.cos(theta) #converts polar cordinates to cartesian but this is sstill in robot fram
		yr = r_mean * math.sin(theta)
		xw = self.x + xr * math.cos(self.yaw) - yr * math.sin(self.yaw) #converts robot x and y to world x and y
		yw = self.y + xr * math.sin(self.yaw) + yr * math.cos(self.yaw)
		return xw, yw #now we return the final positions of the pillars in the world
  
          
		
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
		
		if self.elapsed == 90:
			cv2.imwrite("mymap.png", self.map.map)
#Starts pillar detection			
		pillars = self.detect_pillars(msg) #pillars are still a list of ray clusters from the robot sensors
		for cluster in pillars: #loop over each cluster 
			xw, yw = self.cluster_to_world(cluster, msg) #converts cluster to world cords
			if all(math.hypot(xw - px, yw - py) > 0.8 for px, py in self.known_pillars): #loops over all previous known pillars to determine if this pillar is new, it does it by determining the distance from known pillars to see if it is far enough away to be a new pillar
				self.known_pillars.append((xw, yw)) #becomes a permanant object to the robot "stores pillar location"
        #print(f"Detected pillar at {xw:.2f}, {yw:.2f}") #used for debugging to see if there even was a pillar at a specific cord

		if not self.have_target and self.target_index < len(self.known_pillars): #if we are not moving to a pillar and there exists a know pillar pick the next pillar in the list and move there
			self.tx, self.ty = self.known_pillars[self.target_index]
			self.have_target = True
			
		self.map.show_map()
  
		cmd = Twist()
#Escape timer helps to get the robot away from the current pillar if it is too close
		if self.escape_timer > 0.0:
			cmd.linear.x = -0.2
			cmd.angular.z = 0.6
			self.escape_timer -= 0.1
			self.cmd_pub.publish(cmd)
			return
# go to the pillar
		if self.have_target: #checks if we are driving to a pillar aka has a target
			dx = self.tx - self.x #gives robot the corse to take
			dy = self.ty - self.y
			dist = math.hypot(dx,dy) #used to determine when to stop by calculating how far away the robot is from the pillar
			angle = math.atan2(dy,dx) #gives robot heading in world cords
			err = math.atan2(math.sin(angle - self.yaw), math.cos(angle - self.yaw))
			if dist < 0.55: # determiens how close to stop next to the pillar
				self.visited.append((self.tx, self.ty)) #if we were close enough mark pillar as visited
				self.target_index += 1 #pick next target
				self.have_target = False #make sure I set have target to false so I know I can move on to the next pillar
				self.escape_timer = 1.0 # gets the robot away from the current pillar 
				return
		cmd.linear.x = 0.5 #robot speed
		cmd.angular.z = 1.5 * err
		#else:
			#cmd.angular.z = 0.6
		self.cmd_pub.publish(cmd)


def main(args=None):
	rclpy.init(args=args)
	
	node = Project()
	
	rclpy.spin(node)
	
	node.destroy_node()
	rclpy.shutdown()
	
if __name__ == '__main__':
	main() 
