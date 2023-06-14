import os
import sys
SIMNIBSDIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
# Add the external/lib/win folder to system path so that the dlls there can be found
if sys.platform == 'win32':
    os.environ['PATH'] = os.pathsep.join([
        os.path.join(SIMNIBSDIR, 'external', 'lib', 'win'),
        os.environ['PATH']
    ])
elif sys.platform == 'linux':
    os.environ['PATH'] = os.pathsep.join([
        os.path.join(SIMNIBSDIR, 'external', 'lib', 'linux'),
        os.environ['PATH']
    ])
elif sys.platform == 'darwin':
    os.environ['PATH'] = os.pathsep.join([
        os.path.join(SIMNIBSDIR, 'external', 'lib', 'osx'),
        os.environ['PATH']
    ])
from ._version import __version__
from .mesh_tools import *
from .utils import transformations
from .utils.transformations import *
from .utils import file_finder
from .utils.file_finder import *
from .utils.nnav import localite, softaxic, brainsight
from .utils.mesh_element_properties import ElementTags
from .simulation import sim_struct
from .simulation import fem
from .simulation.run_simnibs import run_simnibs
from .optimization import opt_struct
