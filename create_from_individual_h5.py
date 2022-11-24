import h5py
import os
path = r"/storage/research/data/Jaya-VSM/SumMe/prepared_dataset/layer4_2_relu/"
new_dataset_location = r"/storage/research/data/Jaya-VSM/SumMe/prepared_dataset"
dataset_path = r"/storage/research/data/Jaya-VSM/pytorch-vsumm-reinforce-master/original_dataset_04/eccv16_dataset_summe_google_pool5.h5"
dataset = h5py.File(dataset_path, 'r')
key = dataset.keys()


with h5py.File(new_dataset_location+'/eccv16_dataset_summe_resnet50_layer4_2_relu_img224.h5','w') as f:
    for k in key:
        temp = h5py.File(os.path.join(path + k +'.h5'), 'r')
        av = temp[k]['layer4_2_relu'][...]
        #print(av)
        gt  = dataset[k]['gtscore'][...]
        us = dataset[k]['user_summary'][...]
        cp = dataset[k]['change_points'][...]
        nfps = dataset[k]['n_frame_per_seg'][...]
        nf = dataset[k]['n_frames'][...]
        pi = dataset[k]['picks'][...] 
        ns = dataset[k]['n_steps'][...]
        gts = dataset[k]['gtsummary'][...]
        vn = dataset[k]['video_name'][...]
        f.create_dataset(k + '/features', data=av)
        f.create_dataset(k + '/gtscore', data=gt)
        f.create_dataset(k + '/user_summary', data=us)
        f.create_dataset(k + '/change_points', data=cp)
        f.create_dataset(k + '/n_frame_per_seg', data=nfps)
        f.create_dataset(k + '/n_frames', data=nf)
        f.create_dataset(k + '/picks', data=pi)
        f.create_dataset(k + '/n_steps', data=ns)
        f.create_dataset(k + '/gtsummary', data=gts)
        f.create_dataset(k + '/video_name', data=vn)
