import root_file_io as fio
import read_write_model as sfm_rw
import re
import numpy as np

site_tag = ''

data_root = '/media/siyanlinux/Data/marvin'
target_root = '/home/siyanlinux/Documents/gaussian-splatting'

sfm_tag = 'sfm'
sample_max_target = 'full'

sample_tag = 'train'
marker_label = 85
# sample_tag = 'test'
# marker_label = 59

site_images_dir = fio.createPath(fio.sep, [data_root, site_tag, 'images'])
site_sfm_dir = fio.createPath(fio.sep, [data_root, site_tag, 'sfm'])
site_split_sfm_dir = fio.createPath(fio.sep, [data_root, site_tag, 'split_sfm'])
site_sfm_json_dir = fio.createPath(fio.sep, [data_root, site_tag, 'test_sfm'])

atrium_arcore_seqs=['seq11','seq12','seq13','seq14']
bar_arcore_seqs=['seq14','seq15','seq16','seq17']
church_arcore_seqs=['seq13','seq14','seq15','seq16','seq17']
concourse_arcore_seqs=['seq6','seq7','seq8','seq9','seq10']
square_arcore_seqs=['seq16','seq17','seq18','seq19']
stairs_arcore_seqs=['seq11','seq12','seq13','seq14','seq15']

atrium_test_seqs=['seq5', 'seq6', 'seq13','seq14']
bar_test_seqs=['seq1', 'seq2', 'seq7', 'seq8', 'seq16','seq17']
church_test_seqs=['seq6', 'seq7', 'seq11', 'seq12', 'seq15','seq16','seq17']
concourse_test_seqs=['seq5', 'seq10']
square_test_seqs=['seq7', 'seq8', 'seq9', 'seq10', 'seq11', 'seq18','seq19']
stairs_test_seqs=['seq1','seq3','seq13','seq11']

atrium_test_arcore_camera_id = 419
bar_test_arcore_camera_id = 896
church_test_arcore_camera_id = 469
concourse_test_arcore_camera_id = 839
square_test_arcore_camera_id = 1030
stairs_test_arcore_camera_id = 114

def load_dof(site_dir, cut_off_count, mode='train'):
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

        split_filepth = fio.createPath(fio.sep, [site_split_sfm_dir, scene_tag], 'images' + '_' + mode + '.txt')
        if (fio.file_exist(split_filepth) == False):
            continue
        test_seq_tags = []
        with open(split_filepth, 'r') as split_f:
            test_seq_tags = split_f.readlines()
        if len(test_seq_tags) < 1:
            continue

        dof_image_filepth = fio.createPath(fio.sep, [site_sfm_dir, scene_tag], 'images.txt')
        if fio.file_exist(dof_image_filepth) == False:
            r_= dof_image_filepth.replace('images.txt', 'images.bin')
            if fio.file_exist(r_):
                bin_model = sfm_rw.read_images_binary(r_)
                sfm_rw.write_images_text(bin_model, dof_image_filepth)

        dof_camera_filepth = fio.createPath(fio.sep, [site_sfm_dir, scene_tag], 'cameras.txt')
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

        if mode == 'train':
            dof_point3d_filepth = fio.createPath(fio.sep, [site_sfm_dir, scene_tag], 'points3D.txt')
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
            # if seq_order < 10:
            #     seq_order_str = '0' + seq_order_str
            seq_tag = 'seq' + seq_order_str
            # print(seq_tag)
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
            # from_image_paths = fio.filter_folder(from_image_paths, filter_out=False,filter_text='color')

            total_counter = 0
            for from_imgpth in from_image_paths:
                (imagedir, imagename, imageexit) = fio.get_filename_components(from_imgpth)
                to_imgpth = fio.createPath(fio.sep, [to_image_dir], imagename + '.' + imageexit)
                fio.copy_file(from_imgpth, to_imgpth)
                
                if (cut_off_count_by_scene == -1) == False:
                    if total_counter < cut_off_count_by_scene:
                        total_counter += 1
                    else:
                        break


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
    return images


def get_scene_arcore_seq(scene_tag):
    if (scene_tag == 'atrium'):
        return atrium_arcore_seqs
    elif(scene_tag == 'bar'):
        return bar_arcore_seqs
    elif(scene_tag == 'church'):
        return church_arcore_seqs
    elif(scene_tag == 'concourse'):
        return concourse_arcore_seqs
    elif(scene_tag == 'square'):
        return square_arcore_seqs
    else:
        return stairs_arcore_seqs


def get_scene_arcore_cameraid(scene_tag):
    if (scene_tag == 'atrium'):
        return atrium_test_arcore_camera_id
    elif(scene_tag == 'bar'):
        return bar_test_arcore_camera_id
    elif(scene_tag == 'church'):
        return church_test_arcore_camera_id
    elif(scene_tag == 'concourse'):
        return concourse_test_arcore_camera_id
    elif(scene_tag == 'square'):
        return square_test_arcore_camera_id
    else:
        return stairs_test_arcore_camera_id


def get_scene_testonly_seq(scene_tag):
    if (scene_tag == 'atrium'):
        return atrium_test_seqs
    elif(scene_tag == 'bar'):
        return bar_test_seqs
    elif(scene_tag == 'church'):
        return church_test_seqs
    elif(scene_tag == 'concourse'):
        return concourse_test_seqs
    elif(scene_tag == 'square'):
        return square_arcore_seqs
    else:
        return stairs_test_seqs
    

def load_dof_from_json(site_dir, cut_off_count, mode='test'):
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
        to_scene_dir_sparse_path = fio.createPath(fio.sep, [to_scene_dir, 'sparse', '0'], 'images.txt')

        scene_test_seqs = get_scene_testonly_seq(scene_tag)
        arcore_seqs = get_scene_arcore_seq(scene_tag)

        if fio.file_exist(to_scene_dir_sparse_path):
            fio.delete_file(to_scene_dir_sparse_path)
        
        split_filepth = fio.createPath(fio.sep, [site_dir, scene_tag], 'images' + '_' + mode + '.txt')
        if (fio.file_exist(split_filepth) == False):
            continue
        test_seq_content = []
        with open(split_filepth, 'r') as split_f:
            test_seq_content = split_f.readlines()
        if len(test_seq_content) < 1:
            continue

        with open(to_scene_dir_sparse_path, 'w') as f:
            f.write('# Image list with two lines of data per image:\n')
            f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
            f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
            f.write('# Number of images: ' + str(len(test_seq_content)) + ', mean observations per image: 3433.2455436720143\n')
            image_id = 0
            for test_element in test_seq_content:
                test_element = test_element.strip()
                content = test_element.split(' ')
                if ((len(content) == 8) == False):
                    continue

                target_content = []
                target_content.append(str(image_id))

                target_content += content[4:8]
                target_content += content[1:4]
                
                image_name = content[0]
                seq_tag = image_name.split(fio.sep)[0]
                if (seq_tag not in scene_test_seqs):
                    continue

                camera_id = 1
                if (seq_tag in arcore_seqs):
                    camera_id = get_scene_arcore_cameraid(scene_tag)
                
                target_content.append(str(camera_id))
                target_content.append(image_name)
                target_line = ' '.join(target_content) + '\n'
                f.write(target_line)
                f.write('\n')
                image_id += 1



try:
    if type(sample_max_target) == str:
        cut_off_count = -1
    else:
        cut_off_count = int(sample_max_target)

    load_dof(site_sfm_dir, cut_off_count, mode=sample_tag)
    if (sample_tag) == 'train':
        # load_dof(site_sfm_dir, cut_off_count, mode=sample_tag)
        print("train")
    else:
        load_dof_from_json(site_split_sfm_dir, cut_off_count, mode=sample_tag)       

except Exception as e:
    print(e)
