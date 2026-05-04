import os.path as pth
import numpy as np
from collections import OrderedDict   
from pyreite.OpenMEEGHead import OpenMEEGHead 
from pyreite.data_io import load_tri, load_elecs_dips_txt
from pyreite.EIThelpers import EIT_protocol
from pyreite.optimizers import loss_residuals, jac, hess, jac_hess, \
                               tikhonov, levenberg_marquardt_hessian, \
                               levenberg_marquardt_hessiancheck
from pyreite.colors import printred, printyellow, printgreen, printblue

BASEDIR = pth.dirname(pth.dirname(pth.realpath(__file__)))
DATADIR = pth.join(BASEDIR, 'tests', 'test_data')


def add_noise(x):
    x_shape = x.shape
    x = x.flatten()
    # noise as in Malone 2014
    # proportional noise 
    std_dev_prop = (0.02/100) # 0.02%
    noise_prop = np.array([e*np.random.normal(loc=0.0, scale=np.abs(e)*\
                                                             std_dev_prop) \
                           for e in x])
    # additive noise 
    std_dev_add = 5 / 10**6 # 5 micro Volt
    noise_add = np.random.normal(loc=0.0, scale=std_dev_add)

    #print(x)
    for ii in range(len(x)):
        if x[ii] != 0.0:
            x[ii] += noise_prop[ii] + noise_add
    #print(x)
    x = x.reshape(x_shape)
    return x




def main():

    # Load surface boundary meshes and electrodes of colin
    bnd = {}
    tissues = ['cortex', 'csf', 'skull', 'scalp']
    for tissue in tissues:
        bnd[tissue] = load_tri(pth.join(DATADIR, tissue+'.tri'))
    # Define geometry from inside to outside
    geom = OrderedDict([('cortex', bnd['cortex']), ('csf', bnd['csf']), \
                        ('skull', bnd['skull']), ('scalp', bnd['scalp'])]) 
    # Load electrode positions
    sens = load_elecs_dips_txt(pth.join(DATADIR, 'electrodes_aligned.txt'))





    # change settings only here!
    ###########################################################################

    # Simulate this as the true conductivity of the subject in an experiment
    exp_cond = {'cortex': 0.401, 'csf': 1.65, 'skull': 0.01, 'scalp': 0.365}

    # Choose conductivity values as starting values for our optimization
    cond = {'cortex': 0.301, 'csf': 1.65, 'skull': 0.01, 'scalp': 0.265}
   
    # Don't optimize the following tissue conductivies
    FIXED = []

    ###########################################################################




    # Construct head model of simulated experiment
    experiment = OpenMEEGHead(exp_cond, geom, sens)
    assert experiment.geom.is_nested()
    assert experiment.geom.selfCheck()         

    # EIT measurement protocol, i.e. NeumannDirichlet to voltage operator 
    ND2V = EIT_protocol(num_elec=len(sens), n_freq=1, protocol='all_realistic')

    # Calculate experimental voltage measurements
    V_experiment = experiment.V
    V_experiment = V_experiment.flatten()[ND2V]
    V_experiment = add_noise(V_experiment)


    
    # Construct head model for optimization
    model = OpenMEEGHead(cond, geom, sens, sigma=cond)

    
    # Get voltage difference between exp EIT measurement and our simulation
    dV = loss_residuals(cond, model, V_experiment) 
    Error=0.5*np.nansum(dV**2) # data misfit







    ### Start LMA optimization ###
    step = -1
    print('Error: ', Error)

    while Error > 1e-7:
        step += 1
        print('\n\n\n#########################\n# ITERATIOM STEP NO. %d #\
               \n#########################\n\n' % (step+1))
        old_Error = Error 
        

        ## Calculate material derivatives

        #J = jac_hess(cond, model, V_experiment, fixed=FIXED) 
        #H = hess(cond, model, V_experiment, fixed=FIXED)
        # faster than seperately:
        J, H = jac_hess(cond, model, V_experiment, fixed=FIXED)


        ## Moore-Penrose generalized inverse with Tikhonov regularization:

        # Tikhonov weighting (controls penalty strength of ||x||.
        # As tik_reg_param -> 0, solution -> generalized solution,
        # i.e. no Tikhonov regularization)
        tik_reg_param = 0 # no regularization needed for this low noise level
        # Weight tissue sensitivities (EIT is less sensitive to inner conds)
        Lpr0 = np.identity(len(tissues)) # no weighting



        ## Choose optimizer function

        # Tikhonov
        #delta_x = tikhonov(J, dV, tik_reg_param, Lpr0)

        # LMA with hessian acceleration
        #delta_x = levenberg_marquardt_hessian(J, H, dV, tik_reg_param, Lpr0)
        delta_x = levenberg_marquardt_hessiancheck(J, H, dV,
                                                   tik_reg_param, Lpr0)
        
        diff = OrderedDict([(tissues[i], delta_x[i]) for i in \
                            range(len(tissues))])
        new_cond = OrderedDict([(shell, sigma-diff[shell]) for \
                                 shell, sigma in cond.items()]) 
        for shell, sigma in cond.items():
            if shell in FIXED:
                new_cond[shell] = sigma
                print('Fixed %s' % shell)


        # Get new step data
        dV = loss_residuals(new_cond, model, V_experiment) 
        Error=0.5*np.nansum(dV**2)

        if Error >= old_Error:
            break

            
           
        # Print step results
        delta_x_dict = {t: j for t, j in zip(tissues, delta_x)}
        printgreen('Delta_x '+str(delta_x_dict))
        
        printblue('Experiment_cond:\n'+str(exp_cond))
        print('Cond:\n'+str(cond))
        printyellow('New_cond:\n'+str(new_cond))
        print("Old Error: %f" % old_Error)
        printyellow("New Error: %f\n" % Error)

        

        cond = new_cond



    print('Finished optimization')


if __name__ == '__main__':
    main()
