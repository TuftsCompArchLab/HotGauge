# -*- coding: utf-8 -*-
import os
import stat
import errno
import sys
from uuid import uuid4
from subprocess import Popen, PIPE
import subprocess

import logging
logger = logging.getLogger('RTLDesignFlowScripts.'+__name__)

def pipe_args_to_cmd_stdin(cmd_array, stdin_arr=[], stdin_delim="\n"):
    """
    Runs cmd_array and pipes input from stdin_arr through stdin

    Arguments:
        cmd_array: command line arguments, e.g. ["mv", "one.txt", "two.txt"]
        stdin_arr: values to be piped into the command via stdin
        stdin_delim: delimeter placed between arguments in stdin_arr
    """
    proc = Popen(cmd_array, stdin=PIPE)
    result = proc.communicate(bytes(stdin_delim.join([str(arg) for arg in stdin_arr]), 'UTF-8'))

def terminal_size():
    import fcntl, termios, struct
    th, tw, hp, wp = struct.unpack('HHHH',
                                   fcntl.ioctl(0, termios.TIOCGWINSZ,
                                   struct.pack('HHHH', 0, 0, 0, 0)))
    return tw, th

def run_args_with_cmd_prefix(cmd_prefix, args_list):
    total=len(args_list)
    print_progress_bar(0,total)
    for i,args in enumerate(args_list):
        subprocess.call(" ".join([cmd_prefix] + [args]), shell=True)
        print_progress_bar(i+1,total)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            logger.error("Failed to make directory {}".format(path))
            raise

def make_executable(some_file):
    st = os.stat(some_file)
    return os.chmod(some_file, st.st_mode | stat.S_IEXEC)

def touch_file(some_file):
    return os.utime(some_file, None)

def write_or_update_file(file_name, contents, warn=True, info=False):
    if os.path.isfile(file_name):
        with open(file_name, 'r') as f:
            new_contents = f.read()
            if new_contents == contents: # the file is already up to date
                return True
            elif warn:
                logger.warn('{} has outdated contents and is being overwritten.'.format(file_name))
    elif info:
        logger.info('{} is being created'.format(file_name))

    if mkdir_p(os.path.dirname(file_name)):
        logger.debug("Made directory {} for file {}".format(os.path.dirname(file_name),
                                                             os.path.basename(file_name)))
    with open(file_name, 'w') as f:
        f.write(contents)
    return True


################################## Template filling methods ########################################
def populate_template(template_file, *args, **kwargs):
    if not os.path.isabs(template_file):
        template_file = "{}/{}".format(os.path.dirname(os.path.realpath(__file__)), template_file)
    with open(template_file, "r") as f:
        template = f.read()
    return template.format(*args, **kwargs)

def populate_template_from_instance(inst, template_file=None,*template_args, **template_kwargs):
    if template_file is None:
        err_msg = "Instance of {} must have a template_file attribute".format(inst.__class__)
        assert inst.template_file, err_msg
        template_file = inst.template_file
    template_kwargs['inst'] = inst
    template_kwargs['cls'] = inst.__class__
    contents = populate_template(template_file, *template_args, **template_kwargs)
    return contents

#TODO remove this
def filled_instance_template_to_file(file_name, instance=None, *template_args, **template_kwargs):
    contents = populate_template_from_instance(instance, *template_args, **template_kwargs)
    write_or_update_file(file_name, contents)

def print_progress_bar (iteration, total, prefix='Progress', suffix='Complete', decimals=1, length=None, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar plus prefix/suffix/percent (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if length==None:
        length = terminal_size()[0]-20-len(prefix + suffix)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print('\x1b[2K\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix),)
    # Print New Line on Complete
    if iteration == total:
        print
    sys.stdout.flush()
