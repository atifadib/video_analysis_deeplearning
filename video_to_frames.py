import cv2

def video_to_frames(input_file,output_path='./output/'):
	if(output_path=='./output/'):
		print("Using default path")
	vidcap = cv2.VideoCapture('video.mp4')
	success,image = vidcap.read()
	count = 0
	success = True
	while success:
		success,image = vidcap.read()
		print 'Read a new frame: ', success
		cv2.imwrite(output_path+"frame%d.jpg" % count, image)     # save frame as JPEG file
		count += 1