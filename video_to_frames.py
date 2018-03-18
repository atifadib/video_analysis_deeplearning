import cv2
from os import listdir
import os
from os.path import isfile, join

def video_to_frames(input_file,output_path='./output/'):
	if(output_path=='./output/'):
		print("Using default path")
	vidcap = cv2.VideoCapture(input_file)
	success,image = vidcap.read()
	count = 0
	success = True
	while success:
		success,image = vidcap.read()
		cv2.imwrite(output_path+"frame%d.jpg" % count, image)     # save frame as JPEG file
		count += 1

if(__name__=='__main__'):
		print("Start...")
		mypath='.'
		onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
		for idx,file in enumerate(onlyfiles):
			os.makedirs('./'+str(idx)+'/')
			video_to_frames(file,'./'+str(idx)+'/')
