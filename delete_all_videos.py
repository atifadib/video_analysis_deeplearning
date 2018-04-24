from os import listdir,remove
import os

actions=listdir('.')
for action in actions:
    if(action=='delete_all_videos.py'):
        continue
    else:
        videos_and_folders=listdir('./'+str(action)+'/')
        for f in videos_and_folders:
            if(f.endswith('.avi') or os.path.isfile(f)):
                remove('./'+str(action)+'/'+f)
                print("Removing: ",f)