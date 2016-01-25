import struct
import os
import dt_utils
#GTjoints_view1_seq4_subj2_frame15.txt
#GTjoints_view1_seq4_subj2_frame1
#"/home/coskun/PycharmProjects/data/rnn/train/view1_seq4_subj2_frame15.txt"

# to assign the right array size, or to read the correct number of values in one go, remember:
# GT joints have 54 float elements,
# features have 1024 float elements

dt=dt_utils.laod_pose()
print "ok"