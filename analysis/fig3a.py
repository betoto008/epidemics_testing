import sys
sys.path.append('../library/')
from models import *
from Epi_models import*
from functions import *
import matplotlib.ticker as ticker
import pickle
from tqdm import tqdm

Text_files_path = '../Text_files/Testing_Networks/Stochastic/Networks/'


#----Load data network of contacts----
N = 2000
graphs_names = np.array(['barabasi-albert','watts-strogatz'])
infile_k = open(Text_files_path+'barabasi-albert/k.pck','rb')
k = pickle.load(infile_k)
infile_k.close()
infile_p_k = open(Text_files_path+'barabasi-albert/p_k.pck','rb')
p_k = pickle.load(infile_p_k)
infile_p_k.close()

meanDegree = np.sum(k*p_k)
meanDegree2 = np.sum(k**2*p_k)
T_c = meanDegree/(meanDegree2-meanDegree)
#T_c = 1
gamma = 1/6
tau_SIR = 1/gamma
tau_SEIR = 2*(1/4+gamma)**(-1)

u = np.linspace(0.00005,0.9,100000)
R0_array2 = np.linspace(.75, 6.2, 20)

#Ts_array = np.linspace(T_c, T_c*5, 15)
Ts_array = (1-np.array([np.sum(p_k*(1/(1+(R)/(k**2)))) for R in R0_array2]))
u_sol_array = np.array([u[np.array([np.sum(p_k*k*(1+(i-1)*j)**(k-1)) for i in u])>(np.sum(p_k*k)*u)][-1] for j in Ts_array])
S_SIR = np.array([np.sum(p_k*(1-j+i*j)**k) for (i, j) in zip(u_sol_array, Ts_array)])
#Ts_array = (1-np.array([np.sum(p_k*(1/(1+(b*tau_SEIR)/(k**2)))) for b in (gamma-(((1-R0_array2**2)*((1/4)+gamma))/(4*(1/4))))]))
#u_sol_array = np.array([u[np.array([np.sum(p_k*k*(1+(i-1)*j)**(k-1)) for i in u])>(np.sum(p_k*k)*u)][-1] for j in Ts_array])
S_SEIR = np.array([np.sum(p_k*(1-j+i*j)**(k))*(np.sum(k*p_k*(1-j+i*j)**(k))/(meanDegree)) for (i, j) in zip(u_sol_array, Ts_array)])**1
S_SEIR2 = np.array([np.sum(p_k*(1-j+i*j)**(k))*(np.sum(p_k*(1-j+i*j)**(k))/(1)) for (i, j) in zip(u_sol_array, Ts_array)])**1



ps = [1,0]
sigmas = [1000, 1/4]
R0_array = np.linspace(1, 6.2, 20)
R0s_p1 = np.array([6.0, 4.5, 3.0, 2.0, 1.2])
R0s_p0 = np.array([6.0, 4.5, 3.0, 2.0, 1.2, 0.8])
markersR0 = ['P', '*', '^', 'o', 's', 'X']
betas1 = R0s_p1*gamma
betas2 = R0s_p0*gamma
betas_p = np.array([betas1, betas2], dtype = object)
#R0s_SIR_p = np.array([R0s_p1, (1-np.array([np.sum(p_k*(1/(1+(b*tau_SIR)/k**2))) for b in betas2]))/(1)], dtype = object)
R0s_SIR_p = np.array([R0s_p1, R0s_p0], dtype = object)
#R0s_SEIR_p = np.array([np.sqrt(1-4*(((1/4)*gamma-(1/4)*betas1)/((1/4)+gamma)**2)), (1-np.array([np.sum(p_k*(1/(1+(np.sqrt(1-4*(((1/4)*gamma-(1/4)*b)/((1/4)+gamma)**2)))/(k**2)))) for b in betas2]))/(1)], dtype = object)
#R0s_SEIR_p = np.array([np.sqrt(1-4*(((1/4)*gamma-(1/4)*betas1)/((1/4)+gamma)**2)), (1-np.array([np.sum(p_k*(1/(1+((b*tau_SEIR)/(k*(k))))))/np.sum(p_k) for b in betas2]))/T_c], dtype = object)
R0s_SEIR_p = np.array([np.sqrt(1-4*(((1/4)*gamma-(1/4)*betas1)/((1/4)+gamma)**2)), np.sqrt(1-4*(((1/4)*gamma-(1/4)*betas2)/((1/4)+gamma)**2))], dtype = object)
R0s_all = np.array([R0s_SIR_p, R0s_SEIR_p], dtype = object)

colors_R = plt.cm.Paired(range(7))
colors_p = ['olivedrab', 'indigo']
lines_symbols = []
labels_symbols = []
labels_model = ['SIR', 'SEIR']
fig0, ax0 = plt.subplots(figsize = (10,8), gridspec_kw={'bottom': 0.14,'left':.2})

for l, sigma in enumerate(tqdm(sigmas)):
    fig, ax = plt.subplots(figsize = (10,8), gridspec_kw={'bottom': 0.14,'left':.2})
    if(sigma==1/4):
        ax.set_title('SEIR model', fontsize = 28)
    if(sigma==1000):
        ax.set_title('SIR model', fontsize = 28)
    R0s_p = R0s_all[l]
    for p, betas, R0s, color_p in zip(ps, betas_p, R0s_p, colors_p):
        
        p_ext_array = np.array([])
        i_b = 0
        for beta, R0, color in zip(betas, R0s, colors_R):
            #----Edge Occupancy probability----

            #infile_analized_data = open(Text_files_path+'Stochastic/Networks/barabasi-albert/k_normalization/analized_data_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.pck'%(beta/gamma, sigma, N, p), 'rb')
            infile_analized_data = open(Text_files_path+'barabasi-albert/k_normalization/analized_data_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.pck'%(beta/gamma, sigma, N, p), 'rb')
            nodes_ext , nodes_epi, I_ext, I_epi, max_values, data_ext =  pickle.load(infile_analized_data)
            infile_analized_data.close()
            
            n_epi = len(nodes_epi)
            n_ext = len(nodes_ext)
            
            n_total = n_epi + n_ext
            p_ext_array = np.append(p_ext_array, (n_ext)/n_total)
            
            ax.plot(R0, p_ext_array[-1], marker = markersR0[i_b], color = color_p, ms = 15, linestyle = '')
            if(sigma == 1000):
                ax0.plot(R0, p_ext_array[-1], marker = markersR0[i_b], color = color_p, ms = 15, linestyle = '', alpha = .35)
            if(sigma == 1/4):
                ax0.plot(R0, p_ext_array[-1], marker = markersR0[i_b], color = color_p, ms = 15, linestyle = '')
            i_b +=1
        
        if(p==1.0):
            if(sigma==1000):
                ax.plot(R0_array, ((1/R0_array)),linestyle = 'dotted', linewidth = 3, color = color_p, label = 'Fully-connected')
                ax0.plot(R0_array, ((1/R0_array)),linestyle = 'dotted', linewidth = 3, color = color_p, alpha = .35)
            if(sigma==1/4):
                ax.plot(R0_array, ((1/R0_array)**2),linestyle = 'dashed', linewidth = 3, color = color_p, label = 'Fully-connected')
                ax0.plot(R0_array, ((1/R0_array)**2),linestyle = 'dashed', linewidth = 3, color = color_p, label = 'Fully-connected')
        if(p==0.0):
            if(sigma==1000):
                ax.plot(R0_array2, S_SIR ,linestyle = 'dotted', linewidth = 3, color = color_p, label = r'Network')
                ax0.plot(R0_array2, S_SIR ,linestyle = 'dotted', linewidth = 3, color = color_p, alpha = .35)
            if(sigma==1/4):
                ax.plot(R0_array2, S_SEIR ,linestyle = 'dashed', linewidth = 3, color = color_p, label = r'Network')
                ax.plot(R0_array2, S_SEIR2 ,linestyle = ':', linewidth = 3, color = color_p)
                ax0.plot(R0_array2, S_SEIR ,linestyle = 'dashed', linewidth = 3, color = color_p, label = r'Network')
        
    ax.hlines(1,0.5,6.5, linestyle = 'dashed', color = 'silver', alpha = .4, linewidth = 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.vlines(1,0,1, linestyle = 'dashed', color = 'silver', alpha = .4, linewidth = 1)
    my_plot_layout(ax=ax, xlabel = r'$R_0$', ylabel=r'Extinction probability', x_fontsize = 34, y_fontsize = 34)
    if(sigma==1000):
        ax.set_xlim(0.7,4.6)
        #ax.set_xlim(0,1)
    if(sigma==1/4):
        ax.set_xlim(0.75,2.5)
        #ax.set_xlim(0,1)
    ax.set_ylim(-0.05, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(np.concatenate((lines_symbols,handles)), np.concatenate((labels_symbols,labels)) , fontsize = 24, loc = 1, framealpha=.95)
    fig.savefig('../figures/Stochastic/Networks/barabasi-albert/Prob_ext_'+labels_model[l]+'.png')
    fig.savefig('../figures/Stochastic/Networks/barabasi-albert/pdfs/Prob_ext_'+labels_model[l]+'.pdf')

ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.hlines(1,0.5,6.5, linestyle = 'dashed', color = 'silver', alpha = .4, linewidth = 1)
#ax0.vlines(1,0,1, linestyle = 'dashed', color = 'silver', alpha = .4, linewidth = 1)
ax0.set_xlim(np.min(R0_array2)-.05,np.max(R0_array2)+0.05)
#ax0.set_xlim(0.8,4.6)
my_plot_layout(ax=ax0, xlabel = r'$R_0$', ylabel=r'Extinction probability')
ax0.legend(fontsize = 24, loc = 1, framealpha=.95)
fig0.savefig('../figures/Stochastic/Networks/barabasi-albert/Prob_ext.png')
fig0.savefig('../figures/Stochastic/Networks/barabasi-albert/pdfs/Prob_ext.pdf')

