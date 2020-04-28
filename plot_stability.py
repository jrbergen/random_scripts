from simulation_series_classes import simulationseries, simulation_aunp, simulation_freepol, simulation_stab_freepol
import plotting as plc
from filedirops import load_dict_from_hdf5
import psutil
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from constants import kb
import seaborn as sns
from os.path import join, exists
from os import makedirs
import pandas as pd
import pickle
import plotly.express as px
from matplotlib.ticker import ScalarFormatter, LogFormatter, FormatStrFormatter, FuncFormatter, MultipleLocator, FixedLocator
import sys

def customTickLogFormatter(value, pos):#:, tick_number):
    if value == 0:
        return '0'
    else:
        return f'{value:.0e}'

if __name__ == '__main__':

    # Set process to low-priority so system UI stays responsive
    parent = psutil.Process()
    parent.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)        
        
    testrun = False # Switch for loading small simulation for relatively quick testing
    
    # Set constants/vars
    T = 300.0 # K
    tslist = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180] # Timesteps to be loaded
    plotted_timesteps_lineplots = [5, 100, 160, 180]  # 20, 50, 100, 150] # Specify which timestep sizes should be plotted on lineplots (fs)
    t_max_lineplot = 0.5e-9 # Max plotted physical time for timetraces
    ylim_lplot_scale = 0.8 # Ratio of ylimit to abs(max) and abs(min) encountered value in plotted array
    plotted_time = t_max_lineplot / 1e-15  # ns
    dpi = 400 # Figure pixel density
    figsx, figsy = 12, 10 # Inches
    # rc params for plots
    plt.rc('text', usetex=False)
    plt.rcParams["axes.labelsize"] = 13
    plt.rc('font', family='serif')
            
    # Define simulation folders

    if testrun:
        # Hardcoded directories in this script for now. Oh well..
        simsets = ((r'U:\dumptest_freepol_23_03_2020', 'freepol', 'ssdna'),)
    else:        
        maindir = r"V:\2020_04_07_stability"
        # Get list of simulation directories
        simsetlist = [(join(maindir,
                            rf"2020-04-01-freepol_E5kT_ACF_quarter5e-10dumpinterval_pair_ON_ang_ON_stabilitytest_ts_{ts}"),
                       'stability', 'ssdna',) for ts in tslist]
        simsets = tuple(simsetlist)
    
    supsetdict = {} # Dict for simulations per timestep (each containing several simulations with different initial conditions)
    simsetcnt = 0
    stabilityswitch = True #Switch because parts of this script were pulled out of a more general class for analysis
    
    temp_tengdelta_dir = join(maindir, 'tempvals') # Dir for temporary storage of calculated intermediates (loading the full 250+ GB for every time takes a while..)
    
    # Stupid way of defining a few paths for intermediate storage
    if not exists(temp_tengdelta_dir):
        makedirs(temp_tengdelta_dir)
    path1_tengdelta_pd = join(temp_tengdelta_dir, 'tengdelta_arr.pd')
    path2_tengdeltacat_pd = join(temp_tengdelta_dir, 'tengdelta_arr_cat.pd')
    path3_tengdata_lineplot = join(temp_tengdelta_dir, 'tengdata_lineplot.pkl')
    path4_tengoverpeng = join(temp_tengdelta_dir, 'teng_over_peng.pd')
    path5_temp_tss = join(temp_tengdelta_dir, 'tsnames.pd')
    path6_tengoverpeng_dict_pkl = join(temp_tengdelta_dir, 'teng_over_peng_dict.pkl')
    pathlist = [path1_tengdelta_pd, path3_tengdata_lineplot,
                path4_tengoverpeng,  path6_tengoverpeng_dict_pkl]
                
    stabilityswitch = True
    # Read dump files again if intermediate files don't exist
    readSwitch = True if len([print(f'Non-existing path: {x}') for x in pathlist if not exists(x)]) > 0 else False
    if readSwitch:
        print("Starting readout of dump/log files..")
           for basedir, simtype, dnatype in simsets:
            if 'aunp' in simtype: # Remnants of more general approach
                simclass = simulation_aunp(basedir, simtype, dnatype)
                # simclass.generate_logfileplots()
                # simclass.generate_plots_aunp()
            elif 'freepol' in simtype:
                a = ''
                while a.lower() != 'n' and a.lower() != 'y':
                    a = input('Clear hdf5 files before starting analysis? (Y/N)')
                if a.lower() == 'y':
                    b = ''
                    while b.lower() != 'n' and b.lower() != 'y':
                        b = input('Are you absolutely sure? (Y/N)')
                    if b.lower() == 'y':
                        clearhdf5 = True
                    else:
                        clearhdf5 = False
                else:
                    clearhdf5 = False
                simclass = simulation_freepol(basedir, simtype, dnatype, clearhdf5=clearhdf5)
            elif 'stability' in simtype:
                stabilityswitch = True
                a = ''
                while a.lower() != 'n' and a.lower() != 'y':
                    a = 'n'  # input('Clear hdf5 files before starting analysis? (Y/N)')
                if a.lower() == 'y':
                    b = ''
                    while b.lower() != 'n' and b.lower() != 'y':
                        b = input('Are you absolutely sure? (Y/N)')
                    if b.lower() == 'y':
                        clearhdf5 = True
                    else:
                        clearhdf5 = False
                else:
                    clearhdf5 = False

            supsetdict[tslist[simsetcnt]] = simulation_stab_freepol(basedir, simtype, dnatype,
                                                                    clearhdf5=clearhdf5)
            simsetcnt += 1
    
    if stabilityswitch:      
       if not readSwitch: # Load pickled data if it already exists
            print("Found required paths with pickled intermediates; loading..")
            pd_tengdeltadata = pd.read_pickle(path1_tengdelta_pd)

            pd_tengoverpeng = pd.read_pickle(path4_tengoverpeng)

            with open(path3_tengdata_lineplot, 'rb') as f:
                tengdelta_lineplotdata = pickle.load(f)

            with open(path6_tengoverpeng_dict_pkl, 'rb') as f:
                tengoverpeng_dict = pickle.load(f)

        else:
            print("Did NOT find all paths; recalculating..")
            
            # Initialize some poorly considered/inefficient data structures
            timesteps = []
            tengdeltadata = {
                'timesteps': [],
                'data': [],
                'absdata': [],
                'absdata_log': [],
                'data_E0': [],
                'absdata_E0': [],
            }

            tengdelta_lineplotdata = {
                'timesteps': [],
            }
            pengdeltadata = []
            pengdeltadata_abs = []
            tengoverpeng_dict = {
                'ts':[],
                'means':[],
                'means_log':[],
                'stds':[],
                'stds_log':[],
            }
            tengdelta_arr = []
            
            # Iterate over simulation sets (timstep sizes in this case)
            for i, simclskey in enumerate(supsetdict.keys()): 
                sc = supsetdict[simclskey]  # Load simulation stability analysis class                

                for j, simkey in enumerate(sc.logdict.keys()):  # Iterate over individual simulations
                    
                    # Get total energy
                    teng = sc.logdict[simkey]['TotEng']
                    teng = teng[0]
                    teng = teng[1000:-1000] / (kb * T) # Remove some of the first and last entries in case the simulation crashed to still allow analysis 
                    diffteng = np.diff(teng)
                    tengdeltacut = list(diffteng[1000:-1000])
                    tengdeltacutabs = [abs(x) for x in tengdeltacut]
                    tengdeltadata['data'] += tengdeltacut
                    tengdeltadata['absdata'] += tengdeltacutabs
                    tengdeltadata['absdata_log'] += list(np.log10(np.array(tengdeltacutabs)))
                    E0 = teng[1001]
                    tengdeltadata['data_E0'] += list(np.divide(E0 - tengdeltacut,E0))
                    tengdeltadata['absdata_E0'] += list(np.divide(E0 - np.abs(tengdeltacut), E0))

                    peng = sc.logdict[simkey]['PotEng']
                    peng = peng[0]
                    peng = peng[1000:-1000] / (kb * T)
                    diffpeng = np.diff(peng)
                    pengdeltacut = list(diffpeng[1000:-1000])
                    pengdeltacutabs = [abs(x) for x in pengdeltacut]
                    pengdeltadata += pengdeltacut
                    pengdeltadata_abs += pengdeltacutabs

                    if not j:
                        tsstr = f"{sc.simset_dict[simkey]['timestep'] * 1e15:.0f}"
                        tsint = round(sc.simset_dict[simkey]['timestep'] * 1e15)
                        #print(f'tsint = {tsint}')
                        #print('tsstr = ' + tsstr)
                        tengdelta_lineplotdata[tsstr] = {}
                        tengdelta_lineplotdata['timesteps'].append(tsint)
                        tengdelta_lineplotdata[tsstr]['ydata_teng'] = np.array(tengdeltacut)
                        tengdelta_lineplotdata[tsstr]['ydata_teng_log_abs'] = np.log10(np.abs(np.array(tengdeltacut)))
                        tengdelta_lineplotdata[tsstr]['xdata_teng'] = np.arange(tsint, (len(tengdeltacut) + 1) * tsint,
                                                                           tsint)

                        #[print(f"Tot/pot: {td:.2e}, {pengdeltacut[i]:.2e}, div = {td/pengdeltacut if pengdeltacut !=0 else 99999999999999999:.2e}") for i,td in enumerate(tengdeltacut)]
                        _tengpengdat = np.array(np.divide(np.array(tengdeltacut),np.array(pengdeltacut),
                                                          out=np.zeros_like(tengdeltacut),
                                                          where=(np.array(pengdeltacut) >1e-40)))

                        tengdelta_lineplotdata[tsstr]['ydata_tengpeng'] = _tengpengdat
                        tengdelta_lineplotdata[tsstr]['ydata_tengpeng_log_abs'] = np.log10(np.abs(np.array(_tengpengdat)))
                        tengdelta_lineplotdata[tsstr]['xdata_tengpeng'] = np.arange(tsint, (len(_tengpengdat) + 1) * tsint,
                                                                           tsint)
                                                                           
                    newtimesteps = [round(sc.simset_dict[simkey]['timestep'] * 1e15) for x in range(len(tengdeltacut))]
                    tengdeltadata['timesteps'] += newtimesteps

                t_ad = np.array(tengdeltadata['absdata'])
                p_ad = np.array(pengdeltadata_abs)
                tengoverpeng_abs = np.divide(t_ad,p_ad, out=np.zeros_like(t_ad), where=p_ad>1e-40)
                tengoverpeng_dict['ts'].append(tsint)
                _mean = np.mean(tengoverpeng_abs)
                _std = np.std(tengoverpeng_abs)
                tengoverpeng_dict['means'].append(_mean)
                tengoverpeng_dict['stds'].append(_std)
                tengoverpeng_dict['means_log'].append(np.log10(_mean))
                tengoverpeng_dict['stds_log'].append(np.log10((1/np.log(10))*(_std/_mean)))

                print(f"Processed timestep: {tsint}")
            
            # Store intermediate calculations to disk
            pd_tengdeltadata = pd.DataFrame.from_dict(tengdeltadata)
            pd_tengdeltadata.to_pickle(path1_tengdelta_pd)
            with open(path3_tengdata_lineplot, 'wb') as f:
                pickle.dump(tengdelta_lineplotdata, f)

            pd_tengoverpeng = pd.DataFrame.from_dict(tengoverpeng_dict)
            pd_tengoverpeng.to_pickle(path4_tengoverpeng)
            with open(path6_tengoverpeng_dict_pkl, 'wb') as f:
                pickle.dump(tengoverpeng_dict, f)
    
        # Generate 'stability' plot
        for yaxistype in ('linear',):#, 'log',):
            fig = plt.figure(figsize=(figsx, figsy), dpi=dpi)#plt.subplots(ncols=3, nrows=2, , gridspec_kw={'width_ratios':[3,1]})
            fig.subplots_adjust(top=0.98)

            # Set grids with custom ratios
            outergrid = fig.add_gridspec(2,1, top=0.98)#, hspace=0.2)#, wspace=10)
            ax=[] # List for subplots

            # Initialize grids
            igrid1 = outergrid[0].subgridspec(2, 1)#, wspace=0.0)
            igrid2 = outergrid[1].subgridspec(2, 2, hspace=0.15, wspace=0.15)#, wspace=0.0)

            # Append <etot>/<epot> plot vs timestep size
            ax.append(fig.add_subplot(igrid1[0, :]))

            # Plot errorbars
            ax[0].errorbar(tengoverpeng_dict['ts'], tengoverpeng_dict['means_log'], yerr=tengoverpeng_dict['stds_log'], fmt='none',
                            ecolor='k', capsize = 2.0, capthick=1.0)

            # Plot scatterplot
            ax[0].scatter(tengoverpeng_dict['ts'], tengoverpeng_dict['means_log'])

            # Set labels
            #ax[0].set_ylabel(r"$\left\langle\left|\Delta E_{tot}\right|\right\rangle$ $/$ $\left\langle\left|\Delta E_{pot}\right|\right\rangle$")#fontsize = labelfontsize)
            ax[0].set_ylabel(
                r"$\log_{10}\left(\frac{\left\langle\left|\Delta E_{tot}\right|\right\rangle}{\left\langle\left|\Delta E_{pot}\right|\right\rangle}\right)$")  # fontsize = labelfontsize)

            # Determine yaxis range for yticklabels
            _miny, _maxy, = np.min(tengoverpeng_dict['means']), np.max(tengoverpeng_dict['means'])
            _ndigits = int(round(np.log10((abs(_miny) + abs(_maxy)/2))))
            _minylim, _maxylim = round(_miny * ylim_lplot_scale, -1 * _ndigits),\
                                 round(_maxy * ylim_lplot_scale,-1 * _ndigits)

            # Set ylimit
            ax[0].set_ylim(-2,2)#[-0.07, 0.07])#[_minylim, _maxylim])#

            # Apply proper formatting for y-axis
            #ax[-1].yaxis.set_major_formatter(FuncFormatter(customTickLogFormatter))

            # Append boxplot for delta_etot vs timestep size
            ax.append(fig.add_subplot(igrid1[1,:]))#, sharex = ax[0]))
            print(f"Plotting boxplot: scale = {yaxistype}")
                # outlierprops = dict(markerfacecolor=[0.93,0.25,0.08],
                #                     linestyle = 'none',
                #                     markeredgewidth=0.0)
            ax[1]=sns.boxplot(x='timesteps', y='absdata_log', data=pd_tengdeltadata, showfliers=True)#, flierprops = outlierprops)

            # Set yaxis type (log/lin)
            ax[1].set_yscale(yaxistype)

            # Set labels
            ax[1].set_xlabel("Timestep size (fs)")
            ax[1].set_ylabel(r"$\left|\Delta E_{tot}\right|$ (kT)")

            # Synchronize x-axes for upper plots
            ax[0].set_xlim(pd_tengdeltadata['timesteps'].min() - 0.5,
                           pd_tengdeltadata['timesteps'].max() + 0.5)
            ax[0].xaxis.set_major_locator(FixedLocator([5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]))#MultipleLocator(10)

            #print(f"[int(t) for t in tengoverpeng_dict['ts']] = {[int(t) for t in tengoverpeng_dict['ts']]}")
            #ax[1].set_xticks([int(t) for t in tengoverpeng_dict['ts']])

            #ax[0].xaxis.set_visible(False)
            #ax[0].set_xticks([int(t) for t in tengoverpeng_dict['ts']])

            # Plot lineplots of Etot/Epot
            for i, _ts in enumerate(plotted_timesteps_lineplots):
                #print(f"size pd dict for timestep = {_ts}: {len([x for x in pd_tengdeltadata.timesteps if x == int(_ts)])}")
                ax.append(fig.add_subplot(igrid2[i//int(round(np.sqrt(len(plotted_timesteps_lineplots)))),
                                                 i%int(round(np.sqrt(len(plotted_timesteps_lineplots))))]))

                # Cut off longer simulations (at 1 ns) because energy drift may not scale linear with simulation time.
                no_allowed_points = round(plotted_time/_ts)
                _cury = np.abs(tengdelta_lineplotdata[str(_ts)]['ydata_tengpeng'][:no_allowed_points])
                ax[-1].plot(tengdelta_lineplotdata[str(_ts)]['xdata_tengpeng'][:no_allowed_points] * 1e-6,
                            _cury,
                            label=rf'$\Delta$t = {_ts} fs',
                            linewidth = 0.5)
                print(f'Max of lineplot for ts {_ts} = {np.max(_cury)}')
                print(f'Min of lineplot for ts {_ts} = {np.min(_cury)}')

                # Determine and set axes limits
                _miny, _maxy = np.min(tengdelta_lineplotdata[str(_ts)]['ydata_tengpeng']),\
                    np.max(tengdelta_lineplotdata[str(_ts)]['ydata_tengpeng'])
                _maxdev_order_estim = (abs(_miny) + abs(_maxy)) / 2
                _ndigits = int(round(np.log10(_maxdev_order_estim)))
                _minylim, _maxylim = round(_miny * ylim_lplot_scale, -1 * _ndigits),\
                                     round(_maxy * ylim_lplot_scale, -1 * _ndigits)
                # if i != 1:
                #     #ax[-1].set_ylim([_minylim, _maxylim])
                #     ax[-1].set_ylim([0,100])# _maxylim])
                # else:
                ax[-1].set_ylim([2e-6, 10])#bottom=1e-8)#[0,100])
                ax[-1].set_xlim([0, t_max_lineplot * 1e9])

                #print(f"ylims = {[round(_miny*1.2, -1*_ndigits), round(_maxy*1.2, -1*_ndigits)]}")
                #

                #ax[-1].set_yticks([_miny*1.2,0, _maxy*1.2])
                #ax[-1].ticklabel_format('both',style='sci')
                # Timetrace plots should be linear;
                #   better visualization and logscale cannot display negative values anyway
                ax[-1].set_yscale('log')
                if i%round(np.sqrt(len(plotted_timesteps_lineplots))) == 0:
                    ax[-1].set_ylabel(r"$\frac{\left|\Delta E_{tot}\right|}{\left|\Delta E_{pot}\right|}$", fontsize=13)
                else:
                    ax[-1].yaxis.set_visible(False)

                #LogFormatter())
                #ax[-1].xaxis.set_visible(False)
                if i > 1:
                    ax[-1].set_xlabel("Time (ns)")
                else:
                    ax[-1].xaxis.set_visible(False)


                # Annotate timestep for each timetrace
                ax[-1].legend(loc='upper right', frameon=False, handletextpad=-2.0, handlelength=0)
                #ax[-1].yaxis.set_yticklabels([f'-{_minylim}', '0', f'{_maxylim}'])

            # Only show x-axis for lower subplot
            ax[-1].xaxis.set_visible(True)


            plt.suptitle("Total energy fluctuations for various timestep sizes")
            # #fig.legend(handearles, labels, loc='upper center')
            plt.show()
    