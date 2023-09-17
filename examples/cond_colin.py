import os.path as pth
import numpy as np
from collections import OrderedDict   
from pyreite.OpenMEEGHead import OpenMEEGHead 
from pyreite.data_io import load_tri, load_elecs_dips_txt
from pyreite.material_derivative import EIT_protocol
from pyreite.optimizers import loss_residuals, jac, hess, jac_hess, \
                               tikhonov, levenberg_marquardt_hessian, \
                               levenberg_marquardt_hessiancheck

BASEDIR = pth.dirname(pth.dirname(pth.realpath(__file__)))
DATADIR = pth.join(BASEDIR, 'tests', 'test_data')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def printred(string):
	print(f"{bcolors.FAIL}%s{bcolors.ENDC}" % string)
def printyellow(string):
	print(f"{bcolors.WARNING}%s{bcolors.ENDC}" % string)
def printgreen(string):
	print(f"{bcolors.OKGREEN}%s{bcolors.ENDC}" % string)
def printblue(string):
	print(f"{bcolors.OKBLUE}%s{bcolors.ENDC}" % string)


def is_posdef(M):
    w, _ = np.linalg.eigh(M)
    return True if (w > 0).all() else False

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
    std_dev_add = 5 / pow(10,6) # 5 micro Volt
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
    Error=0.5*np.nansum(pow(dV, 2)) # data misfit


    # Calculate material derivatives
    #J = jac_hess(cond, model, V_experiment, fixed=[]) 
    #H = hess(cond, model, V_experiment, fixed=[])
    J, H = jac_hess(cond, model, V_experiment) #faster than seperately

    # Moore-Penrose generalized inverse with Tikhonov regularization:
    # Tikhonov weighting (controls penalty strength of ||x||.
    # As tik_reg_param -> 0, solution -> generalized solution, i.e. no Tikhonov
    #                                                          regularization)
    tik_reg_param = 0 # no regularization needed for this low noise level
    # Weight tissue sensitivity (EIT is less sensitive for inner conductivies)
    Lpr0 = np.identity(len(tissues)) # no weighting

    # Choose optimizer function
    #jacob = tikhonov(J, dV, tik_reg_param, Lpr0)
    #jacob = levenberg_marquardt_hessian(J, H, dV, tik_reg_param, Lpr0)
    jacob = levenberg_marquardt_hessiancheck(J, H, dV, tik_reg_param, Lpr0)


    ### This part is for analysis only
    V = model.V.flatten()[ND2V]
    n_meas = int((len(sens)*(len(sens)-1))/2)
    n_elecs = len(sens) - 2
    EIT_corr = [np.corrcoef(exp, mod)[0, 1] for exp, mod in \
                zip(V_experiment.reshape(n_meas, n_elecs),
                    V.reshape(n_meas, n_elecs))]
    noser = np.diag(J.conj().T.dot(J))
    condition_jtj = np.linalg.cond(J.conj().T.dot(J))  
    #posdef_jtj = is_posdef(J.conj().T.dot(J))
    condition_lma = np.linalg.cond(J.conj().T.dot(J) + pow(tik_reg_param,2)*Lpr0)
    #posdef_lma = is_posdef(J.conj().T.dot(J) + pow(tik_reg_param,2)*Lpr0)
    condition_h = [np.linalg.cond(H[:,:,i]) for i in range(H.shape[2])]
    posdef_h = [is_posdef(H[:,:,i]) for i in range(H.shape[2])]
    ###



    ### Start LMA optimization ###
    ur_alpha = 1.0 * 2
    step = -1
    steps = {}
    steps[step] = {'experimental_cond': exp_cond, 'starting_cond': cond, \
                   'fixed': FIXED, 'error': Error}
    np.save(pth.join(DATADIR, 'steps_'+str(step)+'.npy'), steps)



    print('Error: ', Error)

    # (Morozov’s discrepancy principle tells us to stop when the output error 
    # first falls below the measurement noise).
    # (np.finfo(np.float32).eps == 1.1920929e-07) #machine precision of float
    # np.finfo(float).eps == 2.220446049250313e-16  # machine precision double
    while Error > 1e-7:
        step += 1
        if not pth.exists(pth.join(DATADIR, 'steps_'+str(step)+'.npy')):
            print('\n\n\n#########################\n# ITERATIOM STEP NO. %d #\
                   \n#########################\n\n' % (step+1))
            old_Error = Error 
            alpha = ur_alpha
            while Error >= old_Error and (alpha>pow(2,-10)):
                alpha/=2
                diff = OrderedDict([(tissues[i], jacob[i]) for i in \
                                    range(len(tissues))])
                new_cond = OrderedDict([(shell, sigma-diff[shell]*alpha) for \
                                         shell, sigma in cond.items()]) 
                for shell, sigma in cond.items():
                    if shell in FIXED:
                        new_cond[shell] = sigma
                        print('Fixed %s' % shell)

                ### for saving (i.e. analysis)
                jacob_dict = {t: j for t, j in zip(tissues, jacob)}
                EIT_corr = [np.corrcoef(exp, mod)[0, 1] for exp, mod in \
                            zip(V_experiment.reshape(n_meas, n_elecs), \
                            V.reshape(n_meas, n_elecs))]
                noser = np.diag(J.conj().T.dot(J))
                condition_jtj = np.linalg.cond(J.conj().T.dot(J))  
                condition_lma = np.linalg.cond(J.conj().T.dot(J) + \
                                pow(tik_reg_param,2)*np.identity(len(tissues)))
                condition_h = [np.linalg.cond(H[:,:,i]) for i in \
                               range(H.shape[2])]
                posdef_h = [is_posdef(H[:,:,i]) for i in range(H.shape[2])]
                ###

               
                # Get new step data
                dV = loss_residuals(new_cond, model, V_experiment) 
                Error=0.5*np.nansum(pow(dV, 2))
                #J = jac_hess(new_cond, model, V_experiment, fixed=[]) 
                #H = hess(new_cond, model, V_experiment, fixed=[])
                J, H = jac_hess(new_cond, model, V_experiment, fixed=[]) 
                jacob = levenberg_marquardt_hessiancheck(J, H, dV, \
                                                         tik_reg_param, Lpr0)
                
               
            # Print step results
            printgreen('Jacobian '+str(jacob_dict))
            
            printblue('Experiment_cond:\n'+str(exp_cond))
            print('Cond:\n'+str(cond))
            printyellow('New_cond after step with step_size '+str(alpha)+ \
                        ':\n'+str(new_cond))
            print("Old Error: %f" % old_Error)
            printyellow("New Error: %f\n" % Error)

            
            update = {'error': Error, 'cond': cond, 'new_cond': new_cond, \
                      'jac': jacob_dict, 'alpha': alpha}
            #          'EIT_corr': EIT_corr, 'noser': noser, 'condition_jtj': \
            #           condition_jtj, 'condition_lma': condition_lma, \
            #          'condition_h': condition_h, 'posdef_h': posdef_h}

            steps[step] = update
            np.save(pth.join(DATADIR,'steps_'+str(step)+'.npy'), {step: update})
        else:
            update = np.load(pth.join(DATADIR, 'steps_'+str(step)+'.npy'), \
                             allow_pickle=True).item()
            Error = update[step]['error']
            new_cond = update[step]['new_cond']

        cond = new_cond



    print('Finished optimization')


if __name__ == '__main__':
    main()
