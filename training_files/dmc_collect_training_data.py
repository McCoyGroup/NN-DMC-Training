import numpy as np
import pyvibdmc as dmc
from pyvibdmc.simulation_utilities import *
from pyvibdmc import potential_manager as pm

pot_dir = '/gscratch/ilahie/rjdiri/Potentials/legacy_mbpol/'
py_file = 'call_mbpol.py'
pot_func = 'call_trimer_zero_e'

tri_pot = pm.Potential(potential_function=pot_func,
                        python_file=py_file,
                        potential_directory=pot_dir,
                        num_cores=28)

init_cds = np.expand_dims(np.load("udu_bohr.npy"),0)



atoms = ['O','H','H'] * 3

factors = np.array([1]*4000 + [0.99972538464]*3999)

myDMC = dmc.DMC_Sim(sim_name="tri_dt10_0",
                    output_folder="hex_training_data",
                    weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                    num_walkers=20000, #number of geometries exploring the potential surface
                    num_timesteps=8000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                    equil_steps=500, #how long before we start collecting wave functions
                    chkpt_every=500, #checkpoint the simulation every "chkpt_every" time steps
                    wfn_every=7999, #collect a wave function every "wfn_every" time steps
                    desc_wt_steps=500, #number of time steps you allow for descendant weighting per wave function
                    atoms=atoms,
                    delta_t=10, #the size of the time step in atomic units
                    potential=tri_pot,
                    start_structures=init_cds, #can provide a single geometry, or an ensemble of geometries
                    DEBUG_save_training_every=5,
                    DEBUG_mass_change={'change_every': 1,
                                       'factor_per_change': factors}
)
myDMC.run()




