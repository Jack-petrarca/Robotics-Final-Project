	# stop from leaving bounds
		edge = .6
		boundary_omega = 1.0
		outward = 0.5
		
		if self.x > 10 - edge and math.cos(self.yaw) > 0.1:
			self.omega = +boundary_omega # turns CCW (left)
		elif self. x < -10 + edge and math.cos(self.yaw) < -0.1:
			self.omega = -boundary_omega # turns CW (right)
		elif self.y > 10 - edge and math.sin(self.yaw) > 0.1:
			self.omega = +boundary_omega
		elif self.y < -10 + edge and math.sin(self.yaw) < -0.1:
			self.omega = -boundary_omega
		else:
		# put collison logic here maybe...
