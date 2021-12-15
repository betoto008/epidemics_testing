import sys
sys.path.append('../library/')
from models import *
from Epi_models import*
from functions import *
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import pickle

Text_files_path = '../Text_files/'


colors_R = plt.cm.Paired(range(7))
models = ['SEIR', 'SIR']
gamma = 1/6
ps=[0.0, 1.0]
sigmas=[1/4, 1000]
u = np.linspace(0.00005,0.9,100000)
tau_SIR = 1/gamma
tau_SEIR = 2*(1/4+gamma)**(-1)

#----Load data network of contacts----
N = 2000
graphs_names = np.array(['barabasi-albert','watts-strogatz'])
infile_k = open(Text_files_path+'Stochastic/Networks/barabasi-albert/k.pck','rb')
k = pickle.load(infile_k)
infile_k.close()
infile_p_k = open(Text_files_path+'Stochastic/Networks/barabasi-albert/p_k.pck','rb')
p_k = pickle.load(infile_p_k)
infile_p_k.close()

meanDegree = np.sum(k*p_k)
meanDegree2 = np.sum(k**2*p_k)
T_c = meanDegree/(meanDegree2-meanDegree)

ns = np.arange(1, 20, 1)

for q, p in enumerate(ps):
        if(p==0.0):
                R0s = np.array([4.5, 3.0, 2.0, 1.2, 0.8])
        if(p==1.0):
                R0s = np.array([4.5, 3.0, 2.0, 1.2])
        for s, sigma in enumerate(sigmas):
                model = models[s]
                #----Plot 1----
                fig, ax = plt.subplots(figsize = (10,8))
                fig2, ax2 = plt.subplots(figsize = (10,8), gridspec_kw={'bottom': 0.14,'left': 0.14})

                R0_array = np.linspace(0.6, 3.0, 20)
                T_array = 1-np.array([np.sum(p_k*(1/(1+((R)/(k*(k)))))) for R in R0_array])
                #T_array = R0_array*T_c
                u_sol_array = np.array([u[np.array([np.sum(p_k*k*(1+(i-1)*T)**(k-1)) for i in u])>(np.sum(p_k*k)*u)][-1] for T in T_array])

                if(sigma==1000): #SIR
                        #Ts = 1- ((meanDegree)/((meanDegree + R0s)))
                        Ts = 1-np.array([np.sum(p_k*(1/(1+(R)/k**2))) for R in R0s])
                        u_sols = np.array([u[np.array([np.sum(p_k*k*(1+(i-1)*T)**(k-1)) for i in u])>(np.sum(p_k*k)*u)][-1] for T in Ts])
                        if(p==1.0):
                                x, y = np.meshgrid(R0_array, ns)
                                z = (1/x)**(y) 
                        if(p==0.0):
                                z0 = np.array([np.sum(k*p_k*(1-j+(j*i))**(k-1))/(np.sum(k*p_k)) for (i, j) in zip(u_sol_array, T_array)])
                                x, y = np.meshgrid(R0_array, ns)
                                x2, y2 = np.meshgrid(z0, ns)
                                z = (x2**(y2))

                if(sigma==1/4): #SEIR
                        Ts = 1-np.array([np.sum(p_k*(1/(1+(np.sqrt(1-4*((sigma*gamma-sigma*b)/(sigma+gamma)**2)))/(k**2)))) for b in R0s*gamma])
                        #Ts = 1-np.array([np.sum(p_k*(1/(1+((R)/(k*(k)))))) for R in R0s2])
                        u_sols = np.array([u[np.array([np.sum(p_k*k*(1+(i-1)*T)**(k-1)) for i in u])>(np.sum(p_k*k)*u)][-1] for T in Ts])
                        if(p==1.0):
                                x, y = np.meshgrid(R0_array, ns)
                                z = (1/x)**(2*y)
                        if(p==0.0):
                                #z0 = np.array([np.sum(k*p_k*(1-j+(j*i))**(k-1))/(np.sum(p_k*k)) for (i, j) in zip(u_sol_array, T_array)])
                                z0 = np.array([np.sum(k*p_k*(1-j+(j*i))**(k-1))/(np.sum(k*p_k)) for (i, j) in zip(u_sol_array, T_array)])
                                x, y = np.meshgrid(R0_array, ns)
                                x2, y2 = np.meshgrid(z0, ns)
                                z = (x2)**(2*y2)

                cs = ax2.contourf(x, y, z, levels = np.linspace(0,1,80), cmap = plt.cm.jet)
                cs2 = ax2.contour(cs, levels=[0.5], colors='k', linestyles = 'dashed', linewidths = 4)

                for r, R0 in enumerate(R0s):

                        beta = R0*gamma

                        #----Epidemic probability as a function of degree of patient zero---- 
                        if(sigma==1000):
                                if(p==1.0):
                                        prob_ext_n = (1/R0)**ns
                                        
                                if(p==0.0):
                                        #R0 = Ts[r]/T_c

                                        u_sol = u_sols[r]
                                        prob_ext_n = (np.sum(k*p_k*(1-Ts[r]+(Ts[r]*u_sol))**(k-1))/(np.sum(k*p_k)))**ns
                                        
                        if(sigma==1/4):
                                if(p==1.0):
                                        R0 = np.sqrt(1-4*(((1/4)*gamma-sigma*beta)/((1/4)+gamma)**2))
                                        prob_ext_n = (1/R0)**(2*ns)
                                        
                                if(p==0.0):
                                        R0 = np.sqrt(1-4*(((1/4)*gamma-sigma*beta)/((1/4)+gamma)**2))
                                        #R0 = Ts[r]/T_c
                                        u_sol = u_sols[r]
                                        prob_ext_n = (np.sum(k*p_k*(1-Ts[r]+(Ts[r]*u_sol))**(k-1))/(np.sum(k*p_k)))**(2*ns)
                                        

                        #data_nodes = np.loadtxt('../../../../Dropbox/Research/Epidemiology_2020/Text_files/Stochastic/Networks/barabasi-albert/k_normalization/stats_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.txt'%(beta/gamma, sigma, N, p))
                        #data_I = np.loadtxt('../../../../Dropbox/Research/Epidemiology_2020/Text_files/Stochastic/Networks/barabasi-albert/k_normalization/ensemble_I_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.txt'%(beta/gamma, sigma, N, p))
                        #data_E = np.loadtxt('../../../../Dropbox/Research/Epidemiology_2020/Text_files/Stochastic/Networks/barabasi-albert/k_normalization/ensemble_E_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.txt'%(beta/gamma, sigma, N, p))
                        #nodes_ext , nodes_epi, I_ext, I_succ, max_values, data_ext = sort_nodes(p=p, beta=beta, sigma=sigma, gamma=gamma, data_I = data_I, data_E = data_E, data_nodes = data_nodes[:,0], upper_limit = 20)
                        infile_analized_data = open(Text_files_path+'Stochastic/Networks/barabasi-albert/k_normalization/analized_data_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.pck'%(beta/gamma, sigma, N, p), 'rb')
                        nodes_ext , nodes_epi, I_ext, I_epi, max_values, data_ext =  pickle.load(infile_analized_data)

                        n_ext = len(nodes_ext)
                        n_total = len(nodes_ext) + len(nodes_epi)
                        prob_ext = n_ext/n_total
                        hist = np.histogram(max_values, bins = np.arange(0, 20, 2), density = False);
                        hist_ext = np.histogram(max_values[data_ext], bins = np.arange(0, 20, 2), density = False);
                        print(n_total, np.cumsum(hist[0])[-1], n_ext, np.cumsum(hist_ext[0])[-1])
                        #a = np.cumsum(hist_ext[0]*(hist_ext[1][1:]-hist_ext[1][:-1]))[-1]
                        #b = np.cumsum(hist[0]*(hist[1][1:]-hist[1][:-1]))[-1]
                        #prob_ext_n_data = 1-(((1-np.cumsum(hist_ext[0][:-1]*(hist_ext[1][1:-1]-hist_ext[1][:-2]))/a)*prob_ext)/(1-np.cumsum(hist[0][:-1]*(hist[1][1:-1]-hist[1][:-2]))/b))
                        prob_ext_n_data = (((1-np.cumsum(hist_ext[0])/n_ext)*prob_ext)/(1-np.cumsum(hist[0])/n_total))
                        ax.plot((hist_ext[1][1:]+1), prob_ext_n_data , '^', color = colors_R[r], ms = 12,  label = r'$R_0=$%.1f'%(R0))
                        ax.plot(ns, prob_ext_n, linewidth = 2, linestyle = '--', color = colors_R[r])

                        for j in np.arange(0, len(prob_ext_n_data), 1):
                                ax2.scatter(R0, (hist_ext[1][1:][j]+1), marker = 's', color = plt.cm.jet(np.linspace(0,1,80))[int(79*prob_ext_n_data[j])], s = 200, edgecolors='k')

                # Plot 1
                ax.hlines(1,0,41, linestyle = '--', color = 'silver')
                ax.set_xticks(np.arange(1,15,2))
                ax.set_xlim(0.5,15)
                ax.set_ylim(-0.05, 1.05)
                my_plot_layout(ax = ax, xlabel = r'$\eta$', ylabel = r'$P(\mathrm{epi}|n_{\mathrm{max}}\geq n)$', yscale = 'linear', x_fontsize = 34, y_fontsize = 34)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(np.concatenate(([],handles)), np.concatenate(([],labels)) , fontsize = 20, loc = 4, framealpha=.95)
                fig.savefig('../figures/Stochastic/Networks/barabasi-albert/Ext_prob_n_'+model+'_p%.1f.png'%(p))
                fig.savefig('../figures/Stochastic/Networks/barabasi-albert/pdfs/Ext_prob_n_'+model+'_p%.1f.pdf'%(p))

                # Plot 2
                my_plot_layout(ax=ax2, xlabel=r'$R_0$', ylabel=r'$\eta$', yscale='log', x_fontsize = 34, y_fontsize = 34)
                if(p==1.0):
                        ax2.set_xlim(1.02, 2.3)
                if(p==0.0):
                        ax2.set_xlim(0.65, 2.3)
                ax2.set_ylim(1.9, 12)
                cbar = fig2.colorbar(cs, ticks=np.linspace(0,1,5))
                cbar.ax.set_ylabel('Extinction probability', fontsize = 25)
                cbar.set_ticklabels(np.linspace(0,1,5))
                cbar.ax.tick_params(labelsize = 25)
                cbar.add_lines(cs2)

                fig2.savefig('../figures/Stochastic/Networks/barabasi-albert/Ext_prob_n_'+model+'_p%.1f_HM.png'%(p))
                fig2.savefig('../figures/Stochastic/Networks/barabasi-albert/pdfs/Ext_prob_n_'+model+'_p%.1f_HM.pdf'%(p))
                
                plt.close()


