
import os
import os.path as osp
import subprocess
import tempfile
from mmcv.utils import requires_executable


@requires_executable('ffmpeg')
def convert_video(in_file, out_file, print_cmd=False, pre_options='', **kwargs):
    'Convert a video with ffmpeg.\n\n    This provides a general api to ffmpeg, the executed command is::\n\n        `ffmpeg -y <pre_options> -i <in_file> <options> <out_file>`\n\n    Options(kwargs) are mapped to ffmpeg commands with the following rules:\n\n    - key=val: "-key val"\n    - key=True: "-key"\n    - key=False: ""\n\n    Args:\n        in_file (str): Input video filename.\n        out_file (str): Output video filename.\n        pre_options (str): Options appears before "-i <in_file>".\n        print_cmd (bool): Whether to print the final ffmpeg command.\n    '
    options = []
    for (k, v) in kwargs.items():
        if isinstance(v, bool):
            if v:
                options.append(''.join(['-', '{}'.format(k)]))
        elif (k == 'log_level'):
            assert (v in ['quiet', 'panic', 'fatal', 'error',
                          'warning', 'info', 'verbose', 'debug', 'trace'])
            options.append(''.join(['-loglevel ', '{}'.format(v)]))
        else:
            options.append(''.join(['-', '{}'.format(k), ' ', '{}'.format(v)]))
    cmd = ''.join(['ffmpeg -y ', '{}'.format(pre_options), ' -i ', '{}'.format(in_file),
                   ' ', '{}'.format(' '.join(options)), ' ', '{}'.format(out_file)])
    if print_cmd:
        print(cmd)
    subprocess.call(cmd, shell=True)


@requires_executable('ffmpeg')
def resize_video(in_file, out_file, size=None, ratio=None, keep_ar=False, log_level='info', print_cmd=False):
    'Resize a video.\n\n    Args:\n        in_file (str): Input video filename.\n        out_file (str): Output video filename.\n        size (tuple): Expected size (w, h), eg, (320, 240) or (320, -1).\n        ratio (tuple or float): Expected resize ratio, (2, 0.5) means\n            (w*2, h*0.5).\n        keep_ar (bool): Whether to keep original aspect ratio.\n        log_level (str): Logging level of ffmpeg.\n        print_cmd (bool): Whether to print the final ffmpeg command.\n    '
    if ((size is None) and (ratio is None)):
        raise ValueError('expected size or ratio must be specified')
    if ((size is not None) and (ratio is not None)):
        raise ValueError('size and ratio cannot be specified at the same time')
    options = {
        'log_level': log_level,
    }
    if size:
        if (not keep_ar):
            options['vf'] = ''.join(
                ['scale=', '{}'.format(size[0]), ':', '{}'.format(size[1])])
        else:
            options['vf'] = ''.join(['scale=w=', '{}'.format(size[0]), ':h=', '{}'.format(
                size[1]), ':force_original_aspect_ratio=decrease'])
    else:
        if (not isinstance(ratio, tuple)):
            ratio = (ratio, ratio)
        options['vf'] = ''.join(
            ['scale="trunc(iw*', '{}'.format(ratio[0]), '):trunc(ih*', '{}'.format(ratio[1]), ')"'])
    convert_video(in_file, out_file, print_cmd, **options)


@requires_executable('ffmpeg')
def cut_video(in_file, out_file, start=None, end=None, vcodec=None, acodec=None, log_level='info', print_cmd=False):
    'Cut a clip from a video.\n\n    Args:\n        in_file (str): Input video filename.\n        out_file (str): Output video filename.\n        start (None or float): Start time (in seconds).\n        end (None or float): End time (in seconds).\n        vcodec (None or str): Output video codec, None for unchanged.\n        acodec (None or str): Output audio codec, None for unchanged.\n        log_level (str): Logging level of ffmpeg.\n        print_cmd (bool): Whether to print the final ffmpeg command.\n    '
    options = {
        'log_level': log_level,
    }
    if (vcodec is None):
        options['vcodec'] = 'copy'
    if (acodec is None):
        options['acodec'] = 'copy'
    if start:
        options['ss'] = start
    else:
        start = 0
    if end:
        options['t'] = (end - start)
    convert_video(in_file, out_file, print_cmd, **options)


@requires_executable('ffmpeg')
def concat_video(video_list, out_file, vcodec=None, acodec=None, log_level='info', print_cmd=False):
    'Concatenate multiple videos into a single one.\n\n    Args:\n        video_list (list): A list of video filenames\n        out_file (str): Output video filename\n        vcodec (None or str): Output video codec, None for unchanged\n        acodec (None or str): Output audio codec, None for unchanged\n        log_level (str): Logging level of ffmpeg.\n        print_cmd (bool): Whether to print the final ffmpeg command.\n    '
    (_, tmp_filename) = tempfile.mkstemp(suffix='.txt', text=True)
    with open(tmp_filename, 'w') as f:
        for filename in video_list:
            f.write(
                ''.join(['file ', '{}'.format(osp.abspath(filename)), '\n']))
    options = {
        'log_level': log_level,
    }
    if (vcodec is None):
        options['vcodec'] = 'copy'
    if (acodec is None):
        options['acodec'] = 'copy'
    convert_video(tmp_filename, out_file, print_cmd,
                  pre_options='-f concat -safe 0', **options)
    os.remove(tmp_filename)
