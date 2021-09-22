import os
import errno
import sys
import time
import random
import multiprocessing
import tqdm
from HotGauge.script_runner.tool_scripts import mkdir_p, pipe_args_to_cmd_stdin, run_args_with_cmd_prefix

import logging
logger = logging.getLogger('RTLDesignFlowScripts.'+__name__)

class ExecutableJobError(EnvironmentError):
    pass

class ExecutableJob(object):
    parallel_options = {"--colsep":" ", "-j":int(0.9*multiprocessing.cpu_count()), "--eta":None}
    # Add/modify options in child class:
    #   import copy
    #   parallel_options = copy.deepcopy(ExecutableJob.parallel_options)
    #   parallel_options["--eta"] = None
    nice = True
    template_file = "executable_job_template.sh"
    def __init__(self, run_path):
        self.run_path = run_path
        mkdir_p(run_path)

    def job_args(self):
        raise NotImplementedError( "ExecutableJob classes must implement a job_args(self) method." )

    @classmethod
    def job_cmd(cls):
        """The value of job_args() is the command by default"""
        return ""

    def input_files(self):
        raise NotImplementedError( "ExecutableJob classes must implement a input_files(self) method." )

    def output_files(self):
        raise NotImplementedError( "ExecutableJob classes must implement a output_files(self) method." )

    def prep_for_run(self):
        """Function that is called before running a job"""
        pass

    @property
    def log_suffix(self):
        return ""

    @staticmethod
    def files_exist(files, strip_path=""):
        missing_files = [f for f in files if not os.path.isfile(f)]
        if len(missing_files)==0:
            return (True, None)
        else:
            msg = "\tMissing Files:\t" + "\t".join(map(lambda f: f.replace(strip_path,""), missing_files))
            return (False, msg)

    @staticmethod
    def files_newer_than_dependancies(files, dependancies, strip_path=""):
        status, msg = ExecutableJob.files_exist(files, strip_path=strip_path)
        if status==False:
            return status, msg
        msgs = []
        for file_ in files:
            output_time = os.path.getmtime(file_)
            newer_dependancies = []
            for dependant_file in dependancies:
                newer_deps = [f_dep for f_dep in dependancies
                                if  os.path.getmtime(f_dep) > output_time]
                if len(newer_deps)>0:
                    stripped_file_names = map(lambda f: f.replace(strip_path,""), newer_deps)
                    file_stripped = file_.replace(strip_path,"")
                    msgs.append("\t{} is older than {}".format(file_stripped, ",".join(stripped_file_names)))
        if len(msgs)==0:
            return True, None
        return False, "\n".join(msgs)

    def outputs_up_to_date(self):
        """Returns False if any of the outputs don't exist or if the inputs are newer"""
        files,deps = self.output_files(), self.input_files()
        return ExecutableJob.files_newer_than_dependancies(files, deps, strip_path=self.run_path)

    def output_status(self):
        """Returns False if any of the outputs are not proper"""
        # Default: assume success. Override as needed
        return (True, None)

    @classmethod
    def run_with_parallels(cls, jobs, shuffle=True):
        """   Runs `parallel {parallel_options} "{nice}? job_cmd() {}" for each job.job_args()`"""
        cmd_array = ['parallel']
        for option, value in cls.parallel_options.items():
            cmd_array.append(str(option))
            if value != None:
                cmd_array.append(str(value))
        job_cmd = cls.job_cmd()
        if cls.nice:
            job_cmd = "nice " + job_cmd
        cmd_array.append(job_cmd)
        cmd_array.append('{}')

        cls._run(cmd_array, jobs, pipe_args_to_cmd_stdin, shuffle=shuffle)

    @classmethod
    def run(cls, jobs):
        job_cmd = cls.job_cmd()
        if cls.nice:
            job_cmd = "nice " + job_cmd
        cls._run(job_cmd, jobs, run_args_with_cmd_prefix)

    @classmethod
    def prep_jobs_for_run(cls, jobs, shuffle=False):
        num_jobs = len(jobs)
        logger.info("Prepping {} {} jobs".format(num_jobs, cls.__name__))
        try:
            pool = multiprocessing.Pool(int(0.75*multiprocessing.cpu_count()))
            tqdm.tqdm(pool.imap_unordered(prep_for_run, jobs), total=len(jobs))
        finally:
            pool.close()
            pool.join()

    @classmethod
    def _run(cls, cmd_array, jobs, run_fn, shuffle=False):
        job_args,jobs_to_run = [],[]
        num_jobs = len(jobs)
        t_start = time.time()
        try:
            pool = multiprocessing.Pool(int(0.75*multiprocessing.cpu_count()))

            if num_jobs > 0:
                logger.info("Checking status of {} {} jobs".format(num_jobs, cls.__name__))
                sys.stdout.flush()

            jobs_to_run = []
            for job in tqdm.tqdm(pool.imap_unordered(check_status, jobs), total=len(jobs)):
                if job:
                    jobs_to_run.append(job)
            total_jobs, num_jobs_to_run = len(jobs), len(jobs_to_run)
            num_jobs_not_run = total_jobs - num_jobs_to_run

            if num_jobs_to_run > 0:
                logger.debug("Getting job arguments for {} {} jobs".format(num_jobs_to_run, cls.__name__))
                sys.stdout.flush()
                job_args = list(tqdm.tqdm(pool.map(get_job_args, jobs_to_run), total=len(jobs_to_run)))

            if num_jobs_not_run > 0:
                logger.info("{}/{} instances of {} do not need to run".format(num_jobs_not_run,
                total_jobs, cls.__name__))
                sys.stdout.flush()

        finally:
            pool.close()
            pool.join()
        if num_jobs_to_run > 0:
            logger.info("Starting {} instances of {}".format(num_jobs_to_run, cls.__name__))
            sys.stdout.flush()
            if shuffle:
                random.shuffle(job_args)
            run_fn(cmd_array, job_args)
        try:
            pool = multiprocessing.Pool(int(0.75*multiprocessing.cpu_count()))

            successful_jobs = num_jobs_not_run
            if num_jobs_to_run > 0:
                logger.debug( "Checking status of {} newly run {} jobs".format(num_jobs_to_run,
                cls.__name__))
                sys.stdout.flush()
                for status, msg in tqdm.tqdm(pool.imap_unordered(check_outputs, [(j,cls) for j in jobs_to_run]), total=len(jobs_to_run)):
                    successful_jobs += status
                    if status == 0:
                        logger.error(msg)

        finally:
            pool.close()
            pool.join()

        t_end = time.time()
        msg = "[I] {}/{} instances of {} are up to date({}seconds elapsed)"
        logger.debug(msg.format(successful_jobs, total_jobs, cls.__name__, t_end-t_start))
        sys.stdout.flush()
        if successful_jobs != total_jobs:
            failed_jobs = total_jobs-successful_jobs
            msg = "{}/{} instances of {} failed.".format(failed_jobs, total_jobs, cls.__name__)
            raise ExecutableJobError(errno.EIO, msg)

def check_outputs(tup):
    job,cls = tup
    failure_msg = "{} FAILED in {}".format(cls.__name__, job.run_path)
    uptd, uptd_msg = job.outputs_up_to_date()
    if not uptd:
        return (False,"{}\n{}".format(failure_msg, uptd_msg))
    status_ok, status_msg = job.output_status()
    if not status_ok:
        return (False, "{}\n{}".format(failure_msg, status_msg))
    return (True,None)
def get_job_args(job):
    return job.job_args()

def prep_for_run(job):
    return job.prep_for_run()

def check_status(job):
    job.prep_for_run()
    status, msg = job.outputs_up_to_date()
    if status==False:
        return job
    status_ok, status_msg = job.output_status()
    if status_ok==False:
        return job
    return None
