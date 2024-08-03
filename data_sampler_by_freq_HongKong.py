import root_file_io as fio
import random

data_root = '/home/siyanlinux/Documents/datasets/HongKong'
sample_max_target = 'full'
sample_rate = 3 # one sample for every 3 frames
target_root = '/home/siyanlinux/Documents/gaussian-splatting'
sample_tag = 'test'

scene_dirs = fio.traverse_dir(data_root, full_path=True, towards_sub=False)
scene_dirs = fio.filter_if_dir(scene_dirs, filter_out_target=False)


def select_random_frames(database, num_frames):
    if num_frames > len(database):
        print("The number of frames to select exceeds the size of the database.")
        return []
    random_indices = random.sample(range(len(database)), num_frames)
    randomly_selected_frames = [database[i] for i in random_indices]
    return randomly_selected_frames


def parse_tag_txt(tt_path, scene_name):
    new_im_dir = fio.createPath(fio.sep, [target_root, 'scene_' + scene_name, 
                                          'seqfreq' + sample_tag + '_' + str(sample_max_target), 'input'])
    if fio.file_exist(new_im_dir):
        # return
        fio.delete_folder(new_im_dir)
    print("saving to...", new_im_dir)
    fio.ensure_dir(new_im_dir)
    contents = []
    (ttdir, ttname, ttext) = fio.get_filename_components(tt_path)
    with open(tt_path, 'r') as f:
        contents = f.readlines()
    im_paths_set = []
    for tt_line in contents:
        combo = tt_line.split(' ')
        if len(combo) <= 0:
            continue
        img_tag = combo[0]
        im_path = fio.createPath(fio.sep, [ttdir], img_tag)
        new_im_path = fio.createPath(fio.sep, [new_im_dir], img_tag.replace(fio.sep, '_'))
        if (fio.file_exist(im_path)) and (fio.check_file_permission(im_path, new_im_path)):
            im_paths_set.append((im_path, new_im_path))
    
    sample_set_max = sample_max_target if len(im_paths_set)/3 > sample_max_target else int(len(im_paths_set)/sample_rate)
    print("Sampling rate: every 3 frames, totally {} frames".format(sample_set_max))
    selected_paths_set = select_random_frames(im_paths_set, sample_set_max)
    if len(selected_paths_set) > 0:
        for sps in selected_paths_set:
            fio.copy_file(sps[0], sps[1])


try:
    for scene_dir_path in scene_dirs:
        sample_cat = fio.createPath(fio.sep, [scene_dir_path], sample_tag + '.txt')
        if fio.file_exist(sample_cat) == False:
            continue
        (scenepnt, scene_name, sceneext) = fio.get_filename_components(scene_dir_path)
        parse_tag_txt(sample_cat, scene_name)

except Exception as e:
    print('Error: ', e)