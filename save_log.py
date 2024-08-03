import root_file_io as fio

root = '/home/xrim/siyan_nerf/3dgs/gaussian-splatting'
scene_name = 'stairs'

raw_folder = fio.createPath(fio.sep, [root, 'data', scene_name])

(rf_dir, tag, rf_ext) = fio.get_filename_components(raw_folder)
input_folder = fio.createPath(fio.sep, [root, 'scene_' + tag])
seq_input_folder = fio.createPath(fio.sep, [input_folder, 'seq1'])
seq_output_folder = fio.createPath(fio.sep, [seq_input_folder, 'output'])
so_files_path = fio.traverse_dir(seq_output_folder, full_path=True, towards_sub=False)

train_log_path = ''
render_log_path = ''
for subfile in so_files_path:
    (filedir, filename, fileext) = fio.get_filename_components(subfile)
    if 'train_log' in filename:
        train_log_path = subfile
        continue
    if filename == 'train':
        subfile = fio.createPath(fio.sep, [subfile, 'ours_30000'])
        sub_files = fio.traverse_dir(subfile, full_path=True, towards_sub=False)
        sub_files = fio.filter_folder(sub_files, filter_out=False, filter_text='render_log')
        if len(sub_files) > 0:
            render_log_path = sub_files[-1]

print('train log path, ', train_log_path)
print('render log path, ', render_log_path)

target_folder = fio.createPath(fio.sep, [root, 'log', scene_name, 'seq1'])
fio.ensure_dir(target_folder)

(traindir, trainname, trainext) = fio.get_filename_components(train_log_path)
(renderdir, rendername, renderext) = fio.get_filename_components(render_log_path)
train_target_path = fio.createPath(fio.sep, [target_folder], trainname + '.' + trainext)
render_target_path = fio.createPath(fio.sep, [target_folder], rendername + '.' + renderext)

print('train target path, ', train_target_path)
print('render target path, ', render_target_path)

fio.copy_file(train_log_path, train_target_path)
fio.copy_file(render_log_path, render_target_path)