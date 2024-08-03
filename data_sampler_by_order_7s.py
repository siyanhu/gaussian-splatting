import root_file_io as fio
import read_write_model as sfm_rw
import re
import random

data_root = '/media/siyanlinux/Data/Hierarchical-Localization'
target_root = '/home/siyanlinux/Documents/gaussian-splatting'

site_tag = '7scenes'

# sfm_tag = 'sfm_superpoint+superglue+depth'
# sample_tag = 'Train'
sfm_tag = '7scenes_sfm_triangulated'
sample_tag = 'Test'
sample_max_target = 'full'

marker_pool = list(range(100))
marker_label = str(random.choice(marker_pool))
marker_label = 59

site_images_dir = fio.createPath(fio.sep, [data_root, site_tag, 'images'])
site_sfm_dir = fio.createPath(fio.sep, [data_root, site_tag, 'enhanced_sfm'])


def load_dof(site_dir, cut_off_count, mode='Train'):
    scene_seq_dirs = fio.traverse_dir(site_dir, full_path=True, towards_sub=False)
    if len(scene_seq_dirs) < 1:
        print("load_dof no scenes at", site_dir)
        return
    
    for scene_path in scene_seq_dirs:
        (scenedir, scene_tag, scene_ext) = fio.get_filename_components(scene_path)
        dof_dir = fio.createPath(fio.sep, [scene_path])
        if fio.file_exist(dof_dir) == False:
            continue

        to_scene_dir = fio.createPath(fio.sep, [target_root, 'scene_' + scene_tag,
                                '_'.join([sample_tag.lower(), str(sample_max_target), 'byorder', str(marker_label)])])
        print(to_scene_dir)
        if fio.file_exist(to_scene_dir):
            # continue
            fio.delete_folder(to_scene_dir)
        fio.ensure_dir(to_scene_dir)

        split_filepth = fio.createPath(fio.sep, [site_dir, scene_tag, 'split_sfm'], mode + 'Split.txt')
        if (fio.file_exist(split_filepth) == False):
            continue
        test_seq_tags = []
        with open(split_filepth, 'r') as split_f:
            test_seq_tags = split_f.readlines()
        if len(test_seq_tags) < 1:
            continue

        dof_image_filepth = fio.createPath(fio.sep, [site_sfm_dir, scene_tag, sfm_tag], 'images.txt')
        if fio.file_exist(dof_image_filepth) == False:
            r_= dof_image_filepth.replace('images.txt', 'images.bin')
            if fio.file_exist(r_):
                bin_model = sfm_rw.read_images_binary(r_)
                sfm_rw.write_images_text(bin_model, dof_image_filepth)

        dof_camera_filepth = fio.createPath(fio.sep, [site_sfm_dir, scene_tag, sfm_tag], 'cameras.txt')
        if fio.file_exist(dof_camera_filepth) == False:
            r_= dof_camera_filepth.replace('cameras.txt', 'cameras.bin')
            if fio.file_exist(r_):
                bin_model = sfm_rw.read_cameras_binary(r_)
                sfm_rw.write_cameras_text(bin_model, dof_camera_filepth)

        if (fio.file_exist(dof_image_filepth) == False) or (fio.file_exist(dof_camera_filepth)==False):
            continue

        to_bin_dir = fio.createPath(fio.sep, [to_scene_dir, 'sparse/0'])
        if fio.file_exist(to_bin_dir):
            fio.delete_folder(to_bin_dir)
        fio.ensure_dir(to_bin_dir)

        if mode == 'Train':
            dof_point3d_filepth = fio.createPath(fio.sep, [site_sfm_dir, scene_tag, sfm_tag], 'points3D.txt')
            if fio.file_exist(dof_point3d_filepth) == False:
                r_= dof_point3d_filepth.replace('points3D.txt', 'points3D.bin')
                if fio.file_exist(r_):
                    bin_model = sfm_rw.read_points3D_binary(r_)
                    sfm_rw.write_points3D_text(bin_model, dof_point3d_filepth)
            if fio.file_exist(dof_point3d_filepth) == False:
                continue
            to_point3d_bin_path = fio.createPath(fio.sep, [to_bin_dir], 'points3D.txt')
            fio.copy_file(dof_point3d_filepth, to_point3d_bin_path)

        to_images_bin_path = fio.createPath(fio.sep, [to_bin_dir], 'images.txt')
        to_camera_bin_path = fio.createPath(fio.sep, [to_bin_dir], 'cameras.txt')
        fio.copy_file(dof_image_filepth, to_images_bin_path)
        fio.copy_file(dof_camera_filepth, to_camera_bin_path)

        cut_off_count_by_scene = -1
        if (cut_off_count == -1) == False:
            cut_off_count_by_scene = int(cut_off_count / len(test_seq_tags))
        # print("cut off for", scene_tag, ' in ', len(test_seq_tags), 'tags:', cut_off_count)
        for raw_seq_tag in test_seq_tags:
            seq_order=int(re.findall(r'\d+', raw_seq_tag)[0])
            seq_order_str = str(seq_order)
            if seq_order < 10:
                seq_order_str = '0' + seq_order_str
            seq_tag = 'seq-' + seq_order_str
            seq_path = fio.createPath(fio.sep, [site_images_dir, scene_tag, seq_tag])
            if fio.file_exist(seq_path) == False:
                continue
            to_scene_seq_dir = fio.createPath(fio.sep, [to_scene_dir, 'images', seq_tag])
            if fio.file_exist(to_scene_seq_dir):
                fio.delete_folder(to_scene_seq_dir)
            fio.ensure_dir(to_scene_seq_dir)

            from_image_dir = fio.createPath(fio.sep, [site_images_dir, scene_tag, seq_tag])
            to_image_dir = fio.createPath(fio.sep, [to_scene_dir, 'images', seq_tag])
            if fio.file_exist(to_image_dir):
                fio.delete_folder(to_image_dir)
            fio.ensure_dir(to_image_dir)

            from_image_paths = fio.traverse_dir(from_image_dir, full_path=True, towards_sub=False)
            from_image_paths = fio.filter_ext(from_image_paths, filter_out_target=False, ext_set=fio.img_ext_set)
            from_image_paths = fio.filter_folder(from_image_paths, filter_out=False,filter_text='color')

            total_counter = 0
            for from_imgpth in from_image_paths:
                (imagedir, imagename, imageexit) = fio.get_filename_components(from_imgpth)
                to_imgpth = fio.createPath(fio.sep, [to_image_dir], imagename + '.color.' + imageexit)
                fio.copy_file(from_imgpth, to_imgpth)
                
                if (cut_off_count_by_scene == -1) == False:
                    if total_counter < cut_off_count_by_scene:
                        total_counter += 1
                    else:
                        break

try:
    if type(sample_max_target) == str:
        cut_off_count = -1
    else:
        cut_off_count = int(sample_max_target)

    load_dof(site_sfm_dir, cut_off_count, mode=sample_tag)

                    

except Exception as e:
    print(e)
