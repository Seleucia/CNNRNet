import platform
import os

def set_list(params):

   params['dataset']=[]
   params['dataset'].extend([-1 for i in range(26)])
   if(platform.node()=="hc"):
       #Category: 3D Object Reconstruction
       params['dataset'][0]=["rgbd_dataset_freiburg3_cabinet","TUM"]
       params['dataset'][1]=["rgbd_dataset_freiburg3_large_cabinet","TUM"]
       params['dataset'][2]=["rgbd_dataset_freiburg3_teddy","TUM"]

       #Category: Handheld SLAM
       params['dataset'][15]=["rgbd_dataset_freiburg3_long_office_household","TUM"]

       #Category: Structure vs. Texture
       params['dataset'][16]=["rgbd_dataset_freiburg3_structure_texture_near","TUM"]

       #ICL
       params['dataset'][17]=["living_room_traj0_frei_png","ICL"]


   if(platform.node()=="milletari-workstation"):
       #Category: 3D Object Reconstruction
       params['dataset'][0]=["rgbd_dataset_freiburg3_cabinet","TUM"]
       params['dataset'][1]=["rgbd_dataset_freiburg3_large_cabinet","TUM"]
       params['dataset'][2]=["rgbd_dataset_freiburg3_teddy","TUM"]

       #Category: Robot SLAM
       params['dataset'][3]=["rgbd_dataset_freiburg2_pioneer_360","TUM"]
       params['dataset'][4]=["rgbd_dataset_freiburg2_pioneer_slam","TUM"]
       params['dataset'][5]=["rgbd_dataset_freiburg2_pioneer_slam2","TUM"]
       params['dataset'][6]=["rgbd_dataset_freiburg2_pioneer_slam3","TUM"]

       #Category: Handheld SLAM
       params['dataset'][7]=["rgbd_dataset_freiburg2_360_hemisphere","TUM"]
       params['dataset'][8]=["rgbd_dataset_freiburg2_coke","TUM"]
       params['dataset'][9]=["rgbd_dataset_freiburg2_desk","TUM"]
       params['dataset'][10]=["rgbd_dataset_freiburg2_dishes","TUM"]
       params['dataset'][11]=["rgbd_dataset_freiburg2_flowerbouquet","TUM"]
       params['dataset'][12]=["rgbd_dataset_freiburg2_flowerbouquet_brownbackground","TUM"]
       params['dataset'][13]=["rgbd_dataset_freiburg2_large_no_loop","TUM"]
       params['dataset'][14]=["rgbd_dataset_freiburg2_large_with_loop","TUM"]
       params['dataset'][15]=["rgbd_dataset_freiburg3_long_office_household","TUM"]

       #Category: Structure vs. Texture
       params['dataset'][16]=["rgbd_dataset_freiburg3_structure_texture_near","TUM"]

       #ICL
       params['dataset'][17]=["living_room_traj0_frei_png","ICL"]
       params['dataset'][18]=["living_room_traj1_frei_png","ICL"]
       params['dataset'][19]=["living_room_traj2_frei_png","ICL"]
       params['dataset'][20]=["living_room_traj3_frei_png","ICL"]
       params['dataset'][21]=["traj0_frei_png","ICL"]
       params['dataset'][22]=["traj1_frei_png","ICL"]
       params['dataset'][23]=["traj2_frei_png","ICL"]
       params['dataset'][24]=["traj3_frei_png","ICL"]

   if(platform.node()=="cmp-comp"):
       params['dataset'].append([])
       params['dataset'][0]=["/home/cmp/projects/data/rgbd_dataset_freiburg3_large_cabinet/","TUM"]
       params['dataset'].append([])
       params['dataset'][1]=["/home/cmp/projects/data/rgbd_dataset_freiburg3_teddy/","TUM"]
       params['dataset'].append([])
       params['dataset'][2]=["/home/cmp/projects/data/rgbd_dataset_freiburg3_cabinet/","TUM"]
       params['dataset'].append([])
       params['dataset'][3]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_coke/","TUM"]
       params['dataset'].append([])
       params['dataset'][4]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_flowerbouquet/","TUM"]
       params['dataset'].append([])
       params['dataset'][5]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_flowerbouquet_brownbackground/","TUM"]
       params['dataset'].append([])
       params['dataset'][6]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_dishes/","TUM"]

   wd=params["wd"]
   dt=os.path.dirname(wd)+"/data/"
   idx=0
   for ds in params["dataset"]:
       if ds ==-1:
           idx+=1
           continue
       params["dataset"][idx][0]=dt+params["dataset"][idx][0]+"/"

       if idx%3!=0 and params['check_mode']==2:
           params["dataset"][idx]=-1
       idx+=1


   return params