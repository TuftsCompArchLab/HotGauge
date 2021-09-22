#!/usr/bin/env python
import sys
import os

import logging.config
import datetime

pwd = os.getcwd()
def dtime_str(time=None):
    time = time if time is not None else datetime.datetime.now()
    return time.strftime('%Y_%m_%d_%H:%M:%S')

SCRIPT_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]
script_start_time = datetime.datetime.now()

FLP_BASE_DIR = os.path.join(pwd, 'floorplans', 'outputs')
EXP_BASE_DIR = os.path.join(pwd, SCRIPT_NAME)

LOG_DIR = os.path.join(EXP_BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_NAME = os.path.join(LOG_DIR, '{}-{}.log'.format(SCRIPT_NAME,dtime_str(script_start_time)))
logging.config.fileConfig('logging.conf', defaults={'logfilename': LOG_NAME}, disable_existing_loggers=False)
LOGGER = logging.getLogger('root.{}'.format(SCRIPT_NAME))
LOGGER.info('Start time: {}'.format(dtime_str()))

OUTPUT_DIR = os.path.join(EXP_BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_level():
    level_str = logging.getLevelName(LOGGER.getEffectiveLevel())
    LOGGER.info("Effective logging level is {}".format(level_str))

LOGGER.log_level = log_level
LOGGER.log_level()

def end_log():
    LOGGER.info('End time: {}'.format(dtime_str()))
LOGGER.log_end = end_log
