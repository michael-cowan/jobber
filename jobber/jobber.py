import utils
import os
import numpy as np
import ase.io

# main path to project folder
# PROJECT_PATH = os.path.join(os.path.expanduser('~'), 'nc_ordering')
# TEST PATH
PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..', 'tests', 'nc_ordering')

DFT_PARAMS = {'279_84': {'nodes': 2,
                         'cores': 18,
                         'bsize': 43.7}}


class Tracker(object):
    """
    """
    def __init__(self):
        self.trackers = ['.running.txt', '.completed.txt', '.failed.txt',
                         '.needscheck.txt']

        self.running = None
        self.running_path = None
        self.completed = None
        self.completed_path = None
        self.failed = None
        self.failed_path = None
        self.needscheck = None
        self.needscheck_path = None

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
                print('%s does not exist.' % name)
                data = []

            # set tracker attribute to list of data (or empty list)
            setattr(self, name, data)

    def run_jobs(self, jobid, xcfunc='PBE'):
        """
        Runs a CP2K job
        """
        pass


class Jobber(Tracker):
    """
    """
    def __init__(self, nc='279_84', dopant='Ag'):
        self.nc = nc
        self.dopant = dopant

        # initialize Tracker info
        super(Jobber, self).__init__()

        # dopant concentrations
        self.dope_concs = np.arange(10, 100, 10)

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

        # get dopant counts
        self.n_dope = (self.n_au * self.dope_concs / 100.).round().astype(int)
        self.actual_conc = (self.n_dope / float(self.n_au)) * 100

        # create job folder structure
        self.init_job_dirs()

    def get_ncid(self, atoms):
        return atoms.get_atomic_symbols()[self.au_indices]
        if len(self.dopant) > 2:
            dopants = [self.dopant[i:i+2]
                       for i in range(0, len(self.dopant), 2)]

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

    def init_job_dirs(self):
        job_dir = os.path.join(self.nc_dir, 'jobs')
        if not os.path.isdir(job_dir):
            os.mkdir(job_dir)

        for c in self.dope_concs:
            c = str(c)
            c_dir = os.path.join(job_dir, c)
            if not os.path.isdir(c_dir):
                os.mkdir(c_dir)
                with open(os.path.join(c_dir, 'INFO'), 'w') as fidw:
                    fidw.write(self.get_info_str(c))

    def get_info_str(self, conc):
        """
        Creates a string of NC info based on
        dopant concentration
        """
        i = np.where(self.dope_concs == int(conc))[0][0]
        txt = self.nc + '\n'

        dopes = [self.dopant[i:i+2] for i in range(0, len(self.dopant), 2)]
        txt += 'Dopant(s): %s\n' % (', '.join(dopes))
        txt += '  % Doped: ' + '%.2f\n' % self.actual_conc[i]
        txt += ' N Dopant: %i\n' % self.n_dope[i]
        txt += '     N Au: %i\n' % (self.n_au - self.n_dope[i])
        return txt


class JobID(object):
    """
    Unique JobID to build path to job and track
    which jobs are running, completed, or failed
    """
    def __init__(self):
        pass

    def __repr__(self):
        return self.value

    def get_path(self):
        pass


if __name__ == '__main__':
    j = Jobber()
