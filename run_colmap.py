import root_file_io as fio

colmap_dir = '/home/siyanlinux/Documents/HongKong/arkit/colmap'
scene_paths = fio.traverse_dir(colmap_dir, full_path=True, towards_sub=False)
scene_paths = fio.filter_if_dir(scene_paths, filter_out_target=False)
for scn_pth in scene_paths:
    image_pnt_dir = fio.createPath(fio.sep, [scn_pth, 'images'])
    seq_paths = fio.traverse_dir(scn_pth, full_path=True, towards_sub=False)
    seq_paths = fio.filter_if_dir(seq_paths, filter_out_target=False)
    for sq_pth in seq_paths:
        image_paths = fio.traverse_dir(sq_pth, full_path=True, towards_sub=False)
        image_files = fio.filter_ext(image_paths, filter_out_target=True, ext_set=fio.img_ext_set)

        if len(image_files) > 0:
            for non_img_pth in image_files:
                fio.delete_file(non_img_pth)

        (seqdir, seqtag, seqext) = fio.get_filename_components(sq_pth) 
        target_sq_pth = fio.createPath(fio.sep, [image_pnt_dir], seqtag)
        fio.move_file(sq_pth, target_sq_pth)