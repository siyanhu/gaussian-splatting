import root_file_io as fio
import read_write_model as sfm_rw
import re
import random

data_root = '/media/siyanlinux/Data/Hierarchical-Localization'
target_root = '/home/siyanlinux/Documents/gaussian-splatting'

site_tag = '7scenes'

sfm_tag = '7Scenes_colmap_poses'
sample_tag = 'Test'

sample_max_target = 'full'

marker_pool = list(range(100))
marker_label = str(random.choice(marker_pool))
marker_label = 59

site_sfm_dir = fio.createPath(fio.sep, [data_root, site_tag, 'enhanced_sfm'])


def load_dof(site_dir, cut_off_count, mode='Train'):
    scene_seq_dirs = fio.traverse_dir(site_dir, full_path=True, towards_sub=False)
    if len(scene_seq_dirs) < 1:
        print("load_dof no scenes at", site_dir)
        return
    
    for scene_path in scene_seq_dirs:
        (scenedir, scene_tag, scene_ext) = fio.get_filename_components(scene_path)
        dof_dir = fio.createPath(fio.sep, [scene_path])
        # if fio.file_exist(dof_dir) == False:
        #     continue

        to_scene_dir = fio.createPath(fio.sep, [target_root, 'scene_' + scene_tag,
                                '_'.join([sample_tag.lower(), str(sample_max_target), 'byorder', str(marker_label)])])
        print(to_scene_dir)
        # seq-02/frame-000240.color.png 0.99176 0.110622 0.0158784 -0.062634 0.708822 0.796624 0.047502 526.22
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

        dof_image_filepth = fio.createPath(fio.sep, [site_sfm_dir, scene_tag, sfm_tag], scene_tag + '_' + sample_tag.lower() + '.txt')
        if fio.file_exist(dof_image_filepth) == False:
            continue

        content = []
        with open (dof_image_filepth, 'r') as fread:
            content = fread.readlines()
        
        new_bin = []
        prefix1 = '# Image list with two lines of data per image:\n'
        prefix2 = '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
        prefix3 = '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
        prefix4 = '# Number of images: ' + str(len(content)) + '\n'
        new_bin += [prefix1, prefix2, prefix3, prefix4]

        for i in range(len(content)):
            rawpose = content[i]
            if (len(rawpose) < 5):
                continue        
            rawpose = rawpose.strip()
            combo = rawpose.split(' ')
            if (len(combo) < 8):
                continue
            IMAGE_ID = str(i)
            NAME = combo[0]
            QW = combo[1] 
            QX = combo[2]
            QY = combo[3]
            QZ = combo[4]
            TX = combo[5]
            TY = combo[6]
            TZ = combo[7]
            CAMERA_ID = str(1)
            newpose = ' '.join([IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME, '\n'])
            new_bin.append(newpose)

        to_bin_dir = fio.createPath(fio.sep, [to_scene_dir, 'sparse/0'])
        to_images_bin_path = fio.createPath(fio.sep, [to_bin_dir], 'images.txt')
        if fio.file_exist(to_images_bin_path):
            to_images_bin_path_temp = fio.createPath(fio.sep, [to_bin_dir], 'images_0.txt')
            fio.move_file(to_images_bin_path, to_images_bin_path_temp)

        with open(to_images_bin_path, 'w') as fwrite:
            fwrite.writelines(new_bin)

try:
    if type(sample_max_target) == str:
        cut_off_count = -1
    else:
        cut_off_count = int(sample_max_target)

    load_dof(site_sfm_dir, cut_off_count, mode=sample_tag)

                    

except Exception as e:
    print(e)
