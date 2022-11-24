# import h5py
# import os
# from utils import Logger
# import sys
# new_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/log/log_h5verify/"
# sys.stdout = Logger(os.path.join(new_path, 'log_h5_file_size_tvsum.txt'))
# org_dataset_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/original_dataset_04/eccv16_dataset_ovp_google_pool5.h5"
# dataset = h5py.File(org_dataset_path, 'r')
# key = dataset.keys()
# vals = dataset.values()
# print(vals)

# for k in key:
#     print(dataset[k].values())
#     print(dataset[k].keys())
#     org_feat = dataset[k]['features'][...]
    # print(f"dataset[k]['gtscore'][...]: {dataset[k]['gtscore'][...]}")
    # # print(f"dataset[k]['user_summary'][...]:{dataset[k]['user_summary'][...]}")
    # # print(f"dataset[k]['change_points'][...]:{dataset[k]['change_points'][...]}")
    # # print(f"dataset[k]['n_frame_per_seg'][...]: {dataset[k]['n_frame_per_seg'][...]}")
    # # print(f"dataset[k]['n_frames'][...]: {dataset[k]['n_frames'][...]}")
    # # print(f"dataset[k]['picks'][...]: {dataset[k]['picks'][...]}") 
    # # print(f"dataset[k]['n_steps'][...]: {dataset[k]['n_steps'][...]}")
    # print(f"dataset[k]['gtsummary'][...]: {dataset[k]['gtsummary'][...]}")
    #print(f"dataset[k]['video_name'][...]: {dataset[k]['video_name'][...]}")




# import h5py
# import os
# from utils import Logger
# import sys
# new_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/log/log_h5verify/"
# sys.stdout = Logger(os.path.join(new_path, 'log_h5_file_size_three.txt'))
# org_dataset_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/original_dataset_04/eccv16_dataset_summe_google_pool5.h5"
# recreated_dataset_path_512 = r"/storage/research/data/Jaya-VSM/SumMe/googlenet/dataset_summe_googlenet_avgpool_512.h5"
# recreated_dataset_path_224 = r"/storage/research/data/Jaya-VSM/SumMe/googlenet/dataset_summe_googlenet_avgpool_224.h5"
# org_dataset = h5py.File(org_dataset_path, 'r')
# rec_dataset_512 = h5py.File(recreated_dataset_path_512, 'r')
# rec_dataset_224 = h5py.File(recreated_dataset_path_224, 'r')
# key = org_dataset.keys()
# vals = org_dataset.values()
# print(key)


# for k in key:
#     print()
#     org_feat = org_dataset[k]['features'][...]
#     print(org_feat[0])
#     rec_feat_512 = rec_dataset_512[k]['features'][...]
#     print(rec_feat_512[0])
#     rec_feat_224 = rec_dataset_224[k]['features'][...]
#     print(rec_feat_224[0])
#     # if org_feat == rec_feat:
#     #     print(f"equal check features : {k}---->yes, org : {org_feat.shape} ####  rec: {rec_feat.shape}")
#     # else:
#     #     print(f"equal check features : {k}---->no, org: {org_feat.shape} #### rec:  {rec_feat.shape}")
#     org_picks = org_dataset[k]['picks'][...]
#     org_picks.sort()
#     print(org_picks, '*****',org_picks.shape)
#     rec_picks_512 = rec_dataset_512[k]['picks'][...]
#     rec_picks_512.sort()
#     print(rec_picks_512, '*****',rec_picks_512.shape)
#     rec_picks_224 = rec_dataset_224[k]['picks'][...]
#     rec_picks_224.sort()
#     print(rec_picks_224, '*****',rec_picks_224.shape)
#     print(f"{org_dataset[k]['video_name'][...]}: org_dataset:{org_dataset[k]['n_frames'][...]}")
#     print(f"{rec_dataset_512[k]['video_name'][...]}: rec_dataset_512:{rec_dataset_512[k]['n_frames'][...]}")
#     print(f"{rec_dataset_224[k]['video_name'][...]}: rec_dataset_224:{rec_dataset_224[k]['n_frames'][...]}")
    # if org_picks == rec_picks:
    #     print(f"equal check picks : {k}---->yes, org: {org_picks.shape} #### rec: {rec_picks.shape}")
    # else:
    #     print(f"equal check picks: {k}---->no, org: {org_picks.shape} #### rec: {rec_picks.shape}")
    # print(dataset[k]['gtscore'][...])
    # print(dataset[k]['user_summary'][...])
    # print(dataset[k]['change_points'][...])
    # print(dataset[k]['n_frame_per_seg'][...])
    # print(dataset[k]['n_frames'][...])
    # print(dataset[k]['picks'][...]) 
    # print(dataset[k]['n_steps'][...])
    # print(dataset[k]['gtsummary'][...])
    # print(dataset[k]['video_name'][...])



import h5py
import os
from utils import Logger
import sys
import numpy as np
new_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/log/log_h5verify"
sys.stdout = Logger(os.path.join(new_path, 'log_h5_file_size_45.txt'))
dataset_path = r"/storage/research/data/Jaya-VSM/SumMe/CNNs/2efficientnet_b7/middle50/4/dataset_summe_efficientnet_b7_features_5_1_add4.h5"
org_dataset_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/original_dataset_04/eccv16_dataset_summe_google_pool5.h5"
org_dataset_tvsum_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/original_dataset_04/eccv16_dataset_tvsum_google_pool5.h5"
re_dataset = h5py.File(dataset_path, 'r')
org_dataset = h5py.File(org_dataset_path, 'r')
org_dataset_tvsum = h5py.File(org_dataset_tvsum_path, 'r')
key = org_dataset.keys()
vals = org_dataset.values()
print(key)
print('***************************')
print(vals)


for k in key:

    print("tvsum explore")
    print(org_dataset_tvsum[k].values())
    print(org_dataset_tvsum[k].keys())
    print('*********************************************')
    re_feat = re_dataset[k]['features'][...]
    print(re_feat)
    print(re_feat.shape)
    re_feat1 = np.squeeze(re_feat)
    print(re_feat1)
    print(re_feat1.shape)
    print('*********************************************')
    org_feat = org_dataset[k]['features'][...]
    print(org_feat)
    print(org_feat.shape)
    break


# path1 = r"/storage/research/data/Jaya-VSM/SumMe/dataset_resnet50/layer4_2_conv1"
# path2 = r"/storage/research/data/Jaya-VSM/SumMe/dataset_resnet50/layer4_2_conv2"

# temp1 = h5py.File(os.path.join(path1 , 'video_1.h5'), 'r')
# av1 = temp1['video_1']['0'][...]
# temp2 = h5py.File(os.path.join(path2 , 'video_1.h5'), 'r')
# av2 = temp2['video_1']['0'][...]

# print(av1==av2)

# new_path = r"/storage/research/data/Jaya-VSM/SumMe/dataset_resnet50_pass1"

# import os
# import sys
# import numpy as np
# from utils import Logger

# sys.stdout = Logger(os.path.join(new_path, 'log_h5_file_size.txt'))

# for filename in os.listdir(new_path):
#     if filename.endswith(".h5") : 
#         file = os.path.join(new_path, filename)
#         print(os.path.join(new_path, filename))
#         temp1 = h5py.File(os.path.join(file), 'r')
#         key = temp1.keys()
#         for k in key:
#             feat = temp1[k]['features'][...]
#             feat = np.array(feat)
#             print(feat.shape)
#             break
#         continue
#     else:
#         continue

