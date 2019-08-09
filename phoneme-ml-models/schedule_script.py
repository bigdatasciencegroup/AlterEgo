import os
import time

def timer(start, end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	return minutes

condition = False
start = time.time()
while True:	
	end = time.time()
	if timer(start, end)==90:
		condition = True
		break

if condition:
	import common_savemodel
	common_savemodel
