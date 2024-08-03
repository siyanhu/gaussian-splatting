import root_file_io as fio

# ground-truth poses:
# 1. sessions/map/trajectories.txt
# 2. sessions/query_val_{hololens,phone}/proc/alignment_trajectories.txt) 
# were obtained using SuperPoint + SuperGlue. 
# For more details, please refer to 
# https://github.com/magicleap/SuperPointPretrainedNetwork 
# and https://github.com/magicleap/SuperGluePretrainedNetwork. 


data_root = '/home/siyanlinux/Documents/datasets/LaMAR/LaMAR_backup'
target_root_gs = '/home/siyanlinux/Documents/datasets/LaMAR/sfm' #invalid due to missing points3d.txt
target_root_colmap = '/home/siyanlinux/Documents/datasets/LaMAR/colmap'

tag = 'query_val_phone' # query_val_hololens, map
scene_dirs = fio.traverse_dir(data_root, full_path=True, towards_sub=False)
scene_dirs = fio.filter_if_dir(scene_dirs, filter_out_target=False)


def copy_session_image(session_dir_paths, target_dir):
    target_dir_image = fio.createPath(fio.sep, [target_dir])
    if fio.file_exist(target_dir_image):
        # continue
        fio.delete_folder(target_dir_image)
    fio.ensure_dir(target_dir_image)
    session_image_dict = {}
    for session_dir in session_dir_paths:
        (sessiondir, sessionname, sessionext) = fio.get_filename_components(session_dir)
        if len(sessionname) < 1:
            continue
        target_dir_image_session = fio.createPath(fio.sep, [target_dir_image, sessionname, 'images'])
        if fio.file_exist(target_dir_image_session):
            # continue
            fio.delete_folder(target_dir_image_session)
        fio.ensure_dir(target_dir_image_session)
        image_dir = fio.createPath(fio.sep, [session_dir, 'images'])
        image_paths = fio.traverse_dir(image_dir, full_path=True, towards_sub=False)
        image_paths = fio.filter_ext(image_paths, filter_out_target=False, ext_set=fio.img_ext_set)
        image_paths = fio.filter_folder(image_paths, filter_out=True, filter_text='._')
        index = 0
        image_dict = {}
        for img_path in image_paths:
            if fio.file_exist(img_path) == False:
                continue
            (imagedir, imagename, imageext) = fio.get_filename_components(img_path)
            target_path = fio.createPath(fio.sep, [target_dir_image_session], 'img' + str(index) + '.' + imageext)
            image_dict[imagename] = index
            fio.copy_file(img_path, target_path)
            index += 1
        session_image_dict[sessionname] = image_dict
    return session_image_dict


def create_sfm_files(camera_fpth, trajectory_fpth, target_dir, session_image_query):
    target_dir = fio.createPath(fio.sep, [target_dir, 'sparse', 'preset'])
    fio.ensure_dir(target_dir)
    if fio.file_exist(camera_fpth) == False:
        return
    output_fpth_c = fio.createPath(fio.sep, [target_dir], 'cameras.txt')
    lines_c = ['# Camera list with one line of data per camera:' + '\n', 
             '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]' + '\n']
    index_c = 1
    camera_dict = {}
    with open(camera_fpth, 'r') as fid_c:
        while True:
            line = fid_c.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split(',')
                if len(elems) < 8:
                    continue
                camera_id = elems[0].strip()
                # print(camera_id) ios_2022-06-20_18.24.49_000/cam_phone_40019949459, ios_2022-06-20_18_40019949459.jpg
                sensor_type = elems[2].strip()
                if 'camera' not in sensor_type:
                    continue
                camera_model = elems[3].strip()
                camera_params = elems[4:-1]
                # if len(camera_params) < 4:
                camera_model = 'SIMPLE_' + camera_model
                camera_params = [x.strip() for x in camera_params]
                line = [str(index_c), camera_model] + camera_params + ['\n']
                lines_c.append(' '.join(line))
                camera_dict[camera_id] = str(index_c)
                index_c += 1
    lines_c.insert(2, '#    Number of cameras: ' + str(index_c) + '\n')
    with open(output_fpth_c, 'w') as fod_c:
        fod_c.writelines(lines_c)

    if fio.file_exist(trajectory_fpth) == False:
        return
    output_fpth_t = fio.createPath(fio.sep, [target_dir], 'images.txt')
    lines_t = ['# Image list with two lines of data per image:' + '\n', 
             '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME' + '\n',
             '#   POINTS2D[] as (X, Y, POINT3D_ID)' + '\n']
    index_t = 1
    with open(trajectory_fpth, 'r') as fid_t:
        while True:
            line = fid_t.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split(',')
                if len(elems) < 9:
                    continue
                camera_id = elems[1].strip()
                if camera_id not in camera_dict:
                    continue

                # print(camera_id) ios_2022-06-20_18.24.49_000/cam_phone_40019949459, ios_2022-06-20_18_40019949459.jpg
                
                incombo = camera_id.split('_')
                # image_name = '_'.join([incombo[0], incombo[1], tl_text, incombo[-1]]) + '.jpg'
                image_name = incombo[-1] + '.jpg'
                session_name = incombo[0]

                camera_index = camera_dict[camera_id]
                camera_params = elems[2:8]
                camera_params = [x.strip() for x in camera_params]
                line = [str(index_t)] + camera_params + [camera_index, image_name, '\n', '\n']
                lines_t.append(' '.join(line))
                index_t += 1
    with open(output_fpth_t, 'w') as fod_t:
        fod_t.writelines(lines_t)

    output_fpth_p = fio.createPath(fio.sep, [target_dir], 'points3D.txt')
    with open(output_fpth_p, 'w') as fod_p:
        pass


for scene_dpth in scene_dirs:
    (scenedir, scenename, sceneext) = fio.get_filename_components(scene_dpth)
    session_dpth = fio.createPath(fio.sep, [scene_dpth, 'sessions', tag, 'raw_data'])
    camera_info_pth = fio.createPath(fio.sep, [scene_dpth, 'sessions', tag], 'sensors.txt')
    trajectory_info_pth = fio.createPath(fio.sep, [scene_dpth, 'sessions', tag], 'trajectories.txt')
    session_dirs = fio.traverse_dir(session_dpth, full_path=True, towards_sub=False)
    session_dirs = fio.filter_if_dir(session_dirs, filter_out_target=False)

    target_dir = fio.createPath(fio.sep, [target_root_colmap, scenename, tag])
    if fio.file_exist(target_dir):
        # continue
        fio.delete_folder(target_dir)
    fio.ensure_dir(target_dir)
    session_image_query = copy_session_image(session_dirs, target_dir)
    create_sfm_files(camera_info_pth, trajectory_info_pth, target_dir, session_image_query)

    sparse_model_path = fio.createPath(fio.sep, [target_dir], 'sparse')
    target_session_dirs = fio.traverse_dir(target_dir, full_path=True, towards_sub=False)
    target_session_dirs = fio.filter_folder(target_session_dirs, filter_out=True, filter_text='sparse')

    for tsd in target_session_dirs:
        target_sparse_model_path = fio.createPath(fio.sep, [tsd], 'sparse')
        fio.copy_folder(sparse_model_path, target_sparse_model_path)
        # image_preset_file = fio.createPath(fio.sep, [target_sparse_model_path, 'preset'], 'images.txt')

        # info = list()
        # with open(image_preset_file, 'r') as f:
        #     info = f.readlines()

        # new_info = list()
        # new_info[0:2] = info[0:2]
        # new_info.append('# Testing \n')
        # for frl in info:
        #     if len(frl) > 5 and frl[0] != "#":
        #         frl = frl.strip()
        #         combo = frl.split(' ')
        #         image_name = combo[-1]
        #         image_path = fio.createPath(fio.sep, [tsd, 'images'], image_name)
        #         if fio.file_exist(image_path) == True:
        #             new_info.append(frl + ' \n\n')

        # fio.delete_file(image_preset_file)
        # with open(image_preset_file, 'w') as f:
        #     f.writelines(new_info)

    fio.delete_folder(sparse_model_path)