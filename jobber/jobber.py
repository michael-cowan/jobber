import utils
import os
import re
import glob
import pathlib
import subprocess
import numpy as np
import ase.io

# main path to project folder
PROJECT_PATH = os.path.join(os.path.expanduser('~'), 'nc_ordering')

# TEST PATH
# PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                             '..', 'tests', 'nc_ordering')

DEFAULT_RUNTYPE = 'PBE'

DEFAULT_PARAMS = {'279_84': {'nodes': 2,
                             'cores': 18,
                             'bsize': 43.7,
                             'run_time': 48}}

RJUST = 20


class Tracker(object):
    """
    """
    def __init__(self):
        self.trackers = ['.running.txt', '.completed.txt', '.failed.txt',
                         '.needscheck.txt']

        # tracker attributes
        self.running = None
        self.completed = None
        self.failed = None
        self.needscheck = None

        # tracker paths
        self.running_path = None
        self.completed_path = None
        self.failed_path = None
        self.needscheck_path = None

        # initialize trackers and tracker paths
        self.init_trackers()

    def init_trackers(self):
        """
        Build tracker and tracker path attributes
        """
        for t in self.trackers:
            # get base name of tracker
            name = t[1:-4]
            pathname = name + '_path'

            # build path string
            path = os.path.join(PROJECT_PATH, t)

            # set tracker path
            setattr(self, pathname, path)

            # try to read in tracker data
            if os.path.isfile(path):
                with open(path, 'r') as fidr:
                    data = fidr.read().strip().split('\n')
            else:
                # print('%s does not exist.' % name)
                data = []

            # set tracker attribute to list of data (or empty list)
            setattr(self, name, data)

    def update_tracker(self, tracker):
        """
        Updates tracker file with new data

        Args:
        tracker (str): name of tracker
                       - running, completed, failed, or needscheck
        """
        with open(getattr(self, tracker + '_path'), 'w') as fidw:
            fidw.write('\n'.join(getattr(self, tracker)))

    def setup_job(self, jobid):
        """
        Sets up a CP2K job with default options
        """
        nodes = str(DEFAULT_PARAMS[jobid.nc]['nodes'])
        cores = str(DEFAULT_PARAMS[jobid.nc]['cores'])
        bsize = str(DEFAULT_PARAMS[jobid.nc]['bsize'])
        run_time = str(DEFAULT_PARAMS[jobid.nc]['run_time'])

        if not os.path.isdir(jobid.path):
            raise NotADirectoryError(jobid.path)

        cwd = os.getcwd()

        os.chdir(jobid.path)

        sj = ['sj', jobid.xyz, '-n', nodes, '-c', cores, '-t', run_time,
              '-f', jobid.runtype, '-s', bsize, '--title', str(jobid),
              '--donotcenter']

        # TODO: remove this and test on Comet
        # print(' '.join(sj))
        # return

        # run setup job script - sj
        subprocess.call(sj)

        # move back to CWD
        os.chdir(cwd)

    def submit_job(self, jobid):
        """
        Runs a CP2K job
        """

        cwd = os.getcwd()
        os.chdir(jobid.path)

        # TODO: uncomment this and test on Comet
        slurm_id = subprocess.check_output(['sbatch', jobid.slurm]).split()[-1]

        # save slurm id as empty file
        open(os.path.join(jobid.path, slurm_id), 'w').close()

        self.running.append(str(jobid))
        print('Submitted:'.rjust(RJUST) + ' %s' % str(jobid))

        os.chdir(cwd)


class Jobber(Tracker):
    """
    """
    def __init__(self, nc='279_84', dopant='Ag'):
        self.nc = nc
        self.dopant = dopant
        self.dopant_list = [self.dopant[i:i+2]
                            for i in range(0, len(self.dopant), 2)]

        # initialize Tracker info
        super(Jobber, self).__init__()

        # main NC directory
        self.nc_dir = os.path.join(PROJECT_PATH, nc)

        # get optimized nc while ensuring NC folder is properly structured
        self.nc_opt = None
        self.core_opt = None
        self.shell_opt = None
        if not self.get_opt_nc():
            raise ValueError("Invalid NC given. Make sure NC folder has been"
                             "made and includes optimized NC (in auopt)")

        # tag nc atom order
        self.nc_opt.set_tags(list(range(len(self.nc_opt))))

        # get au indices
        self.au_indices = np.where(self.nc_opt.numbers == 79)[0]

        # get core indices (match based on position of core_opt atoms)
        self.core_indices = np.zeros(len(self.au_indices))
        for i, c in enumerate(self.core_opt):
            self.core_indices[i] = np.where((abs(c.position -
                                                 self.nc_opt.positions) < 0.1)
                                            .all(axis=1))[0][0]

        # get structure info
        self.n_au = len(self.au_indices)
        self.n_s = (self.nc_opt.numbers == 16).sum()
        self.n_core = len(self.core_opt)
        self.n_shellint = self.shell_opt.info['nshellint']

        # create job directory if not made
        self.job_dir = os.path.join(self.nc_dir, 'jobs')
        if not os.path.isdir(self.job_dir):
            os.mkdir(self.job_dir)

    def get_ncid(self, atoms):
        """
        Generates a unique ordering NC ID
        """
        return atoms.get_atomic_numbers()[self.au_indices]

    def get_opt_nc(self):
        """
        Get optimized NC xyz (read in as ASE atoms object)
        Ensures NC folder is properly structured
        Returns:
            - False if any data is not found
        """
        if not os.path.isdir(self.nc_dir):
            return False

        auopt = os.path.join(self.nc_dir, 'auopt')
        if not os.path.isdir(auopt):
            return False

        # look for optimized nc, core, and shell xyz files
        # read in each xyz as an ASE atoms object
        for i, name in enumerate([self.nc, 'core', 'shell']):
            xyz = os.path.join(auopt, '%s_opt.xyz' % name)
            if not os.path.isfile(xyz):
                return False
            # NC attribute should use generic 'nc_opt' name
            if i == 0:
                name = 'nc'
            setattr(self, name + '_opt', ase.io.read(xyz))

        return True

    def get_info_str(self, conc, n_dope, actual_conc):
        """
        Creates a string of NC info based on
        dopant concentration
        """
        txt = self.nc + '\n'
        txt += 'Dopant(s): %s\n' % (', '.join(self.dopant_list))
        txt += '  % Doped: ' + '%.2f\n' % actual_conc
        txt += ' N Dopant: %i\n' % n_dope
        txt += '     N Au: %i\n' % (self.n_au - n_dope)
        return txt

    def init_conc_dir(self, conc):
        """
        Initialize folder for jobs with <conc> dopant concentration

        Returns:
        conc_dir (str): path to dopant conc. folder
        n_dope (int): actual number of dopants
        """
        # get number of dopants
        n_dope = int(round(self.n_au * conc / 100.))

        # get actual dopant percentage (in % units)
        actual_conc = (n_dope / float(self.n_au)) * 100.

        # directory is actual concentration of dopant to 1 decimal point
        # e.g. 10.212% = dirname '10_2'
        actual_str = str(round(actual_conc, 1))
        integer = actual_str[:actual_str.index('.')]
        decimal = actual_str[-1]
        conc_base = '%02i_%s' % (int(integer), decimal)
        conc_dir = os.path.join(self.job_dir, conc_base)

        # create dope_conc directory and add INFO file
        if not os.path.isdir(conc_dir):
            os.mkdir(conc_dir)
            with open(os.path.join(conc_dir, 'INFO'), 'w') as fidw:
                fidw.write(self.get_info_str(conc, n_dope, actual_conc))

        # return concentration folder path and number of dopants
        return conc_dir, n_dope

    def gen_orderings(self, dope_conc, n, only_core=False, run=False,
                      runtype=DEFAULT_RUNTYPE, verbose=True):
        """
        Generates <n> random chemical orderings with a given <dope_conc>
        """
        # get indices where dopant can be added
        indices = self.core_indices if only_core else self.au_indices

        # initialize dopant conc directory if necessary
        conc_dir, n_dope = self.init_conc_dir(dope_conc)

        # get basename of conc directory
        conc_dirname = os.path.basename(conc_dir)

        # read in previous ncids
        prev_ncids_path = os.path.join(conc_dir, '.prev_ncids.npz')
        if os.path.isfile(prev_ncids_path):
            prev_ncids = np.load(prev_ncids_path)['arr_0']
            if prev_ncids.shape[0] != np.unique(prev_ncids, axis=0).shape[0]:
                print('Not all orderings are unique :(')
        else:
            prev_ncids = None

        # build array of random orderings
        orderings = utils.multidimensional_shifting(
                        num_samples=int(1.5 * n),
                        sample_size=n_dope,
                        elements=indices.copy())

        # create new atoms objects and save them to a job directory
        n_made = 0
        new_ncids = []
        jobids = [None] * n
        for o in orderings:
            if n_made == n:
                break

            # generate jobid and build folder
            jobid = JobID(self, conc_dirname, runtype=runtype)

            # create new atoms object and add dopant
            new_atoms = self.nc_opt.copy()
            syms = np.array(new_atoms.get_chemical_symbols())
            # NOTE: only supports single dopant at the moment
            # maybe I can use hyphenated concentration string for more metals
            syms[o] = self.dopant
            new_atoms.set_chemical_symbols(syms)

            # get unique ncid for dopant
            ncid = self.get_ncid(new_atoms)

            # ensure ncid has not already been generated
            # if it has, continue on to next random ordering
            if isinstance(prev_ncids, np.ndarray):
                if ((prev_ncids == ncid).all(axis=1)).any():
                    print('Ordering match found.. go buy a lottery ticket!')
                    continue

            # build folder for job
            jobid.build_folder()

            # set jobid to atoms object and save as xyz
            new_atoms.info['jobid'] = str(jobid)
            new_atoms.write(jobid.xyz_path)

            # write jobid file that contains ncid
            np.save(os.path.join(jobid.path, '%s.npy' % str(jobid)), ncid)

            # let user know jobid has been initialized
            if verbose and jobid.is_initialized():
                print('Initialized:'.rjust(RJUST) + ' %s' % str(jobid))

            # setup job
            self.setup_job(jobid)

            # submit job
            if run:
                self.submit_job(jobid)

            # add ncid to list of accepted new_ncids
            new_ncids.append(ncid)

            # add jobid to list
            jobids[n_made] = jobid

            # ensure only <n> orderings are created
            n_made += 1

        # update and save ncids
        if isinstance(prev_ncids, np.ndarray):
            all_ncids = np.concatenate((prev_ncids, new_ncids))
        else:
            all_ncids = np.array(new_ncids)
        np.savez_compressed(prev_ncids_path, all_ncids)

        # if submitted jobs, update running tracker
        if run:
            self.update_tracker('running')

        # return JobIDs
        return jobids


class JobID(object):
    """
    Unique JobID to build path to job and track
    which jobs are running, completed, or failed
    """
    def __init__(self, jobber=None, conc_dirname=None,
                 runtype=DEFAULT_RUNTYPE, jobid_str=None):
        # jobid can be initialized by arguments or
        # a previously generated jobid string
        self.nc_dir = None
        self.nc = None
        self.dopant = None
        self.conc_dirname = None
        self.ordering_id = None
        self.runtype = None
        if jobid_str is not None:
            self.parse_jobid_str(jobid_str)
        else:
            self.parse_args(jobber, conc_dirname)

        # build jobid
        self.value = '-'.join([self.nc, self.dopant, self.conc_dirname,
                               self.ordering_id, self.runtype])

        # path to job directory based on jobid
        self.path = os.path.join(self.nc_dir,
                                 'jobs',
                                 *self.value.split('-')[2:])

        # xyz name for atoms object and path to xyz
        self.xyz = self.nc + '_' + self.ordering_id + '.xyz'
        self.xyz_path = os.path.join(self.path, self.xyz)

        # get job submission file names
        base = self.nc.split('_')[0]
        self.slurm = 'slurm_%s.sl' % base
        self.input = 'input_%s.inp' % base
        self.output = 'output_%s.out' % base

    def __repr__(self):
        return self.value

    def __eq__(self, compare):
        """
        Test to see if two JobIds are the same
        """
        for p in ['nc', 'nc_dir', 'dopant', 'dope_concname',
                  'ordering_id', 'runtype', 'value']:
            if getattr(self, p) != getattr(compare, p):
                return False
        return True

    def parse_args(self, jobber, conc_dirname, runtype=DEFAULT_RUNTYPE):
        """
        Parses arguments to get job info
        """
        self.nc_dir = jobber.nc_dir
        self.nc = jobber.nc
        self.dopant = jobber.dopant
        self.conc_dirname = str(conc_dirname)
        self.ordering_id = self.get_ordering_id()
        self.runtype = runtype

    def parse_jobid_str(self, jobid_str):
        """
        Parses jobid string to populate attributes
        """
        parts = jobid_str.split('-')
        if len(parts) != 5:
            raise ValueError("Invalid JobId string given.")

        self.nc = parts[0]
        self.nc_dir = os.path.join(PROJECT_PATH, self.nc)
        self.dopant = parts[1]
        self.conc_dirname = parts[2]
        self.ordering_id = parts[3]
        self.runtype = parts[4]

    def build_folder(self):
        pathlib.Path(self.path).mkdir(parents=True)

    def get_ordering_id(self):
        """
        Finds the next ordering id number
        """
        path = os.path.join(self.nc_dir, 'jobs', self.conc_dirname)

        # attempt to find all previous ordering id folders
        last_found = glob.glob(os.path.join(path, '[0-9]' * 6))

        # return next id or 1 if no folders were found
        i = 1 if not last_found else int(os.path.basename(max(last_found))) + 1

        # return as a six digit string padded with 0s
        return '%06i' % i

    def is_initialized(self):
        """
        Returns True of jobid folder has been properly initialized
        """
        return utils.files_exist(self.path, [self.xyz, str(self) + '.npy'])

    def is_setup(self):
        """
        Returns true if a CP2K job has been setup
        """
        return utils.files_exist(self.path, [self.slurm, self.input])

    def is_running(self):
        """
        TODO: search sacct results to see if jobid is running
        """
        return utils.files_exist(self.path, [self.output])

    def is_completed(self):
        has_output = utils.files_exist(self.path, [self.output])

        # if not has... lol
        if not has_output:
            return False

        # see if output file ends with CP2K completion string
        last_line = '  **** **  *******  **  PROGRAM STOPPED IN'
        with open(self.output, 'rb') as fidr:
            fidr.seek(-160, 2)
            return fidr.readlines()[-1].startswith(last_line)


if __name__ == '__main__':
    j = Jobber()

    # for conc in range(10, 100, 10):
    jobids = j.gen_orderings(50, n=2, run=True)

    jobid = jobids[0]
    print(jobid)
    print(jobid.is_initialized())
    print(jobid.is_running())
