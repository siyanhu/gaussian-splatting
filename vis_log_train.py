import sys
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

import root_file_io as fio
import pandas as pd


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Visualising" + args.model_path + ' for ' + args.source_path)

    print(args.source_path)


# def parse_train_log(log_path):
#     content = []
#     with open(log_path, 'r') as f:
#         content = f.readlines()
#     log_list = []
#     for line in content:
#         line = line.replace('\n', '')
#         combo = line.split(' ')
#         if len(combo) < 8:
#             continue
#         iter =int(combo[1].replace(']', ''))
#         tag = combo[3].replace(':', '')
#         l1 = float(combo[5])
#         psnr = float(combo[7].replace('\n', ''))
#         combo_dict = {'iteration': iter, 'tag': tag, 'l1_loss': l1, 'psnr': psnr}
#         log_list.append(combo_dict)
#     log_df = pd.DataFrame(log_list)
#     return log_df


# def parse_render_log(log_path):
#     content = []
#     with open(log_path, 'r') as f:
#         content = f.readlines()
#     log_list = []
#     # [INDEX 0] TimeElapse:9
#     for line in content:
#         line = line.replace('\n', '')
#         combo = line.split(' ')
#         if len(combo) < 3:
#             continue
#         iter =int(combo[1].replace(']', ''))
#         tag = combo[2].replace(':', '')
#         timeElap = int(tag[-1])
#         combo_dict = {'iteration': iter, 'tag': 'render', 'renderElapse': timeElap}
#         log_list.append(combo_dict)
#     log_df = pd.DataFrame(log_list)
#     log_df['avgElapse'] = log_df['renderElapse'].cumsum() / range(1, log_df.shape[0] + 1)
#     return log_df


# root = '/Users/siyanhu/GitHub/stat/3dgs_stat'
# scene_dir_list = fio.traverse_dir(root, full_path=True, towards_sub=False)

# scene_dict = {}
# for scene_path in scene_dir_list:
#     (scenepnt, scene_name, scene_ext) = fio.get_filename_components(scene_path)
#     scene_name = scene_name.replace('scene_', '')
#     log_files = fio.traverse_dir(scene_path, full_path=True, towards_sub=False)
#     train_log_path = ''
#     render_log_path = ''
#     for log_path in log_files:
#         (logdir, log_name, logext) = fio.get_filename_components(log_path)
#         if 'train_log' in log_name:
#             train_log_path = log_path
#             continue
#         if 'render_log' in log_name:
#             render_log_path = log_path
#             continue
#     if len(train_log_path) > 0 and len(render_log_path) > 0:
#         train_df = parse_train_log(train_log_path)
#         render_df = parse_render_log(render_log_path)
#         scene_dict[scene_name] = (train_df, render_df)


# stat_list = []
# for scene_name, logdfs in scene_dict.items():
#     (train_df, render_df) = logdfs

#     ax = train_df['psnr'].plot(figsize=(16,6))
#     ax.set_title('PSNR per Iteration for Scene ' + scene_name)
#     fig = ax.get_figure()
#     save_path = fio.createPath(fio.sep, [root, scene_name], 'PSNR.png')
#     fig.savefig(save_path)
#     fig.clf()

#     ax = train_df['l1_loss'].plot(figsize=(16,6))
#     ax.set_title('L1 Loss per Iteration for Scene ' + scene_name)
#     ax.set_ylim(0,1.0)
#     fig = ax.get_figure()
#     save_path = fio.createPath(fio.sep, [root, scene_name], 'L1Loss.png')
#     fig.savefig(save_path)
#     fig.clf()

#     ax = render_df.avgElapse.plot(figsize=(16,6))
#     ax.set_title('Everage Rendering TimeElapse after Iterations for Scene ' + scene_name)
#     fig = ax.get_figure()
#     save_path = fio.createPath(fio.sep, [root, scene_name], 'AvgRenderElapse.png')
#     fig.savefig(save_path)
#     fig.clf()

#     iter1_l1 = train_df.loc[train_df['iteration'] == 1]['l1_loss'].values[-1]
#     iter14999_l1 = train_df.loc[train_df['iteration'] == 14999]['l1_loss'].values[-1]
#     iter29999_l1 = train_df.loc[train_df['iteration'] == 29999]['l1_loss'].values[-1]

#     iter1_psnr = train_df.loc[train_df['iteration'] == 1]['psnr'].values[-1]
#     iter14999_psnr = train_df.loc[train_df['iteration'] == 14999]['psnr'].values[-1]
#     iter29999_psnr = train_df.loc[train_df['iteration'] == 29999]['psnr'].values[-1]

#     render_row, render_column = render_df.shape
#     first_index = 1
#     mid_index = int(render_row/2)
#     last_index = render_row - 1
#     iter1_render_avg = render_df.loc[render_df['iteration'] == first_index]['avgElapse'].values[-1]
#     iter14999_render_avg = render_df.loc[render_df['iteration'] == mid_index]['avgElapse'].values[-1]
#     iter29999_render_avg = render_df.loc[render_df['iteration'] == last_index]['avgElapse'].values[-1]

#     scene_stat = {'scene': scene_name,
#                   'train_image_num': render_row,
#                   'train_loss_iter1': iter1_l1, 'train_loss_iter14999': iter14999_l1, 'train_loss_iter29999': iter29999_l1,
#                   'train_psnr_iter1': iter1_psnr, 'train_psnr_iter14999': iter14999_psnr, 'train_psnr_iter29999': iter29999_psnr,
#                   'render_avgElapse_iter1': iter1_render_avg, 'render_avgElapse_itermid': iter14999_render_avg, 'render_avgElapse_iterlast': iter29999_render_avg
#                   }
#     stat_list.append(scene_stat)

# stat_df = pd.DataFrame(stat_list)
# stat_df_savePath = fio.createPath(fio.sep, [root], str(tio.current_timestamp()) + '_stat.csv')
# fio.save_df_to_csv(stat_df, stat_df_savePath)