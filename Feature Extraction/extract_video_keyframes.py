# # # -*- coding: utf-8 -*-
# # """process_videos.ipynb"""


"""To filter out unnecessary videos"""
import pandas as pd
import shutil
df_all_data = pd.read_csv('/home/development/apoorvan/AN_MTP/cikm/data/mustard++_sarcasm_detection.csv')
all_keys = list(set(df_all_data['SCENE']))
print(len(all_keys))
path = "/home/development/apoorvan/AN_MTP/cikm/data/final_datasets/mustard++"
for fname in all_keys:
  shutil.move(path+"/utterance/"+fname+'_u.mp4',"/home/development/apoorvan/AN_MTP/cikm/data/final_datasets/final_utterance/"+fname+'_u.mp4')
  shutil.move(path+"/context/"+fname+'_c.mp4',"/home/development/apoorvan/AN_MTP/cikm/data/final_datasets/final_context/"+fname+'_c.mp4')

"""The following code uses the katna library to extract key frames from our videos"""


from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os

# For windows, the below if condition is must.
if __name__ == "__main__":

  # initialize video module
  vd = Video()

  # number of images to be returned
  no_of_frames_to_returned = 1024
  commonpath = '/home/development/apoorvan/AN_MTP/cikm/data/final_datasets/'
  
  # initialize diskwriter to save data at desired location
  diskwriter = KeyFrameDiskWriter(location=commonpath+"test_context_frames")
  # Video file path
  video_file_path = os.path.join(commonpath+"final_context_videos/1_S12E22_004_c.mp4")

  print(f"Input video file path = {video_file_path}")

  # extract keyframes and process data with diskwriter
  vd.extract_video_keyframes(
       no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
       writer=diskwriter
  )