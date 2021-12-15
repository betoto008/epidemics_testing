import sys
sys.path.append('../library/')
from models import *
from Epi_models import*
from functions import *
import networkx as nx
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import pickle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick



Text_files_path = '../Text_files/Testing_Networks/'


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

sample_sizes = np.array([int((N/100)*1.875), int((N/100)*3.125), int((N/100)*5.0)])
gamma = 1/6
sigmas=[1/4]
models = ['SEIR', 'SIR']
ps = np.array([0.0, 1.0])
R0s = np.array([1.2, 2.0, 3.0, 4.5])
R0s2 = np.array([0.8, 1.2, 2.0, 3.0, 4.5])
colors_m = ['darkgreen', 'darkblue', 'darkred']
colors = ['green', 'blue', 'red']
tau_SEIR = 2*(1/4+gamma)**(-1)

n50_0 = np.transpose(np.tile(np.array([5, 3, 2, 1, 1]), (3,1)))
n50_1 = np.transpose(np.tile(np.array([4, 2, 1, 1]), (3,1)))


for s, sigma in enumerate(sigmas):

	Betas = np.array([R0s2*gamma,R0s*gamma], dtype=object)
	#print('Betas:', Betas)
	#R0_Ns = (1-np.array([np.sum(p_k*(1/(1+R/k**2))) for R in R0s2]))/T_c
	R0_Ns = R0s2
	R0_Es = np.sqrt(1-4*((sigma*gamma-sigma*Betas[1])/(sigma + gamma)**2))
	R0_ENs = np.sqrt(1-4*((sigma*gamma-sigma*Betas[0])/(sigma + gamma)**2))
	#R0_ENs = (1-np.array([np.sum(p_k*(1/(1+(b*tau_SEIR)/k**2))) for b in R0s2*gamma]))/T_c

	lambdas = ((-sigma-gamma)/(2)) + (1/2)*np.sqrt((sigma-gamma)**2 + 4*sigma*Betas[1]) #exponential growth rates

	ests = (1/(lambdas)).astype(int)
	if(sigma==1/4):
	    est_Ns = (1/((R0_ENs-1)*gamma)).astype(int)
	if(sigma==1000):
	    est_Ns = (1/((R0_Ns-1)*gamma)).astype(int)

	Ts_Total = np.array([5*est_Ns,5*ests], dtype="object") #for R0 = [1.2, 1.5, 3.0, 4.5] use [160, 35, 20] and half of it if when sigma=1000
	print('Times:',Ts_Total)

	model = models[s]
	i_p = 0
	for p, Ts_total, betas in zip(ps, Ts_Total, Betas):
		if(p==0.0):
			n50 = n50_0
		if(p==1.0):
			n50 = n50_1
		i_b = 0
		fig, ax = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.12, 'left': 0.14})
		fig2, ax2 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.12, 'left': 0.14})
		fig3, ax3 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.12, 'left': 0.14})
		fig4, ax4 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.12, 'left': 0.14})
		fig5, ax5 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.12, 'left': 0.14})
		fig6, ax6 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.12, 'left': 0.14})

		avg_t_uniform = np.array([])
		avg_cs_uniform = np.array([])
		avg_t_aposteriori = np.array([])
		avg_cs_aposteriori = np.array([])

		if(sigma == 1000):
			if(p==0.0):
				R0s = R0_Ns
		if(sigma==1/4):
			if(p==0.0):
				R0s = R0_ENs
			if(p==1.0):
				R0s = R0_Es


		for beta, T_total in zip(betas, Ts_total):
	        
			print('p:', p, 'sig:', sigma, 'R0:', beta/gamma, 'T:', T_total)
			i_m = 1
			for m, color, color_m in zip(sample_sizes, colors, colors_m):
				data_sampling_uniform = np.loadtxt(Text_files_path+'Sampling/Networks/barabasi-albert/uniform/k_normalization/likelihood/sampling_stats_R0%.1f_sigma%.1f_N%d_p%.1f_m%d_barabasi-albert.txt'%(beta/gamma,sigma,N,p,m))
				data_sampling_uniform_times = data_sampling_uniform[data_sampling_uniform[:,0]!=1000,0]
				data_sampling_uniform_cs = data_sampling_uniform[data_sampling_uniform[:,0]!=1000,5]

				data_sampling_aposteriori = np.loadtxt(Text_files_path+'Sampling/Networks/barabasi-albert/aposteriori/k_normalization/likelihood/sampling_stats_R0%.1f_sigma%.1f_N%d_p%.1f_m%d_barabasi-albert.txt'%(beta/gamma,sigma,N,p,m))
				data_sampling_aposteriori_times = data_sampling_aposteriori[data_sampling_aposteriori[:,0]!=1000,0]
				data_sampling_aposteriori_cs = data_sampling_aposteriori[data_sampling_aposteriori[:,0]!=1000,5]

				avg_t_uniform = np.append(avg_t_uniform, np.mean(data_sampling_uniform_times))
				avg_cs_uniform = np.append(avg_cs_uniform, np.mean(data_sampling_uniform_cs))

				avg_t_aposteriori = np.append(avg_t_aposteriori, np.mean(data_sampling_aposteriori_times))
				avg_cs_aposteriori = np.append(avg_cs_aposteriori, np.mean(data_sampling_aposteriori_cs))
				i_m+=1
			i_b += 1

		avg_t_uniform = np.reshape(avg_t_uniform, (np.size(Ts_total), np.size(sample_sizes)))
		avg_cs_uniform = np.reshape(avg_cs_uniform, (np.size(Ts_total), np.size(sample_sizes)))
		avg_t_aposteriori = np.reshape(avg_t_aposteriori, (np.size(Ts_total), np.size(sample_sizes)))
		avg_cs_aposteriori = np.reshape(avg_cs_aposteriori, (np.size(Ts_total), np.size(sample_sizes)))

		if(p==0.0):
			vmin_t_0 = np.min(((np.max(avg_t_aposteriori/avg_t_uniform)**(-1), np.min(avg_t_aposteriori/avg_t_uniform))))
			vmax_t_0 = np.max((np.max(avg_t_aposteriori/avg_t_uniform), np.min(avg_t_aposteriori/avg_t_uniform)**(-1)))
			vmin_cs_0 = np.min((np.max(avg_cs_aposteriori/avg_cs_uniform)**(-1), np.min(avg_cs_aposteriori/avg_cs_uniform)))
			vmax_cs_0 = np.max((np.max(avg_cs_aposteriori/avg_cs_uniform), np.min(avg_cs_aposteriori/avg_cs_uniform)**(-1)))

		sns.heatmap(avg_t_uniform, vmin = 0, vmax=np.max(avg_t_uniform), ax = ax, cmap=plt.cm.twilight, center = 0, cbar = True, cbar_kws={'label': r'$T_U$'}, linewidths=.5)
		sns.heatmap(avg_cs_uniform/n50, vmin = np.min((np.max(avg_cs_uniform/n50)**(-1), np.min(avg_cs_uniform/n50))), vmax=np.max((np.max(avg_cs_uniform/n50), np.min(avg_cs_uniform/n50)**(-1))), ax = ax2, cmap=plt.cm.seismic, center = 1, cbar = True, cbar_kws={'label': r'$n_U/n^*$'}, linewidths=.5)
		sns.heatmap(avg_t_aposteriori, vmin = 0, vmax=np.max(avg_t_aposteriori), ax = ax3, cmap=plt.cm.twilight, center = 0, cbar = True, cbar_kws={'label': r'$T_k$'}, linewidths=.5)
		sns.heatmap(avg_cs_aposteriori/n50, vmin = np.min((np.max(avg_cs_aposteriori/n50)**(-1), np.min(avg_cs_aposteriori/n50))), vmax=np.max((np.max(avg_cs_aposteriori/n50), np.min(avg_cs_aposteriori/n50)**(-1))), ax = ax4, cmap=plt.cm.seismic, center = 1, cbar = True, cbar_kws={'label': r'$n_k/n^*$'}, linewidths=.5)
		sns.heatmap(avg_t_aposteriori/avg_t_uniform, vmin = vmin_t_0, vmax= vmax_t_0, ax = ax5, cmap=plt.cm.seismic, center = 1, cbar = True, cbar_kws={'label': r'$T_k/T_{U}$'}, linewidths=.5)
		sns.heatmap(avg_cs_aposteriori/avg_cs_uniform, vmin = vmin_cs_0, vmax= vmax_cs_0, ax = ax6, cmap=plt.cm.seismic, center = 1, cbar = True, cbar_kws={'label': r'$n_k/n_{U}$'}, linewidths=.5)


		my_plot_layout(ax=ax, xlabel=r'Sample size (%)', ylabel=r'$R_0$', xscale='linear', x_fontsize = 34, y_fontsize = 34)
		ax.set_xticks([.5, 1.5, 2.])
		ax.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(sample_sizes/N*100))
		ax.set_yticks(np.array([g + 0.5 for g in np.arange(len(betas))]))
		ax.set_yticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		cbar = ax.collections[0].colorbar
		cbar.ax.tick_params(labelsize=18)
		ax.figure.axes[-1].yaxis.label.set_size(30)
		fig.savefig('../figures/Sampling/Networks/barabasi-albert/avg_t_uniform_'+model+'_p%.1f_L.pdf'%(p))

		plt.close(fig)

		my_plot_layout(ax=ax2, xlabel=r'Sample size (%)', ylabel=r'$R_0$', xscale='linear', x_fontsize = 34, y_fontsize = 34)
		ax2.set_xticks([.5, 1.5, 2.5])
		ax2.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(sample_sizes/N*100))
		ax2.set_yticks(np.array([g + 0.5 for g in np.arange(len(betas))]))
		ax2.set_yticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		cbar = ax2.collections[0].colorbar
		cbar.ax.tick_params(labelsize=18)
		ax2.figure.axes[-1].yaxis.label.set_size(30)
		fig2.savefig('../figures/Sampling/Networks/barabasi-albert/avg_cs_uniform_'+model+'_p%.1f_L.pdf'%(p))
		plt.close(fig2)

		my_plot_layout(ax=ax3, xlabel=r'Sample size (%)', ylabel=r'$R_0$', yscale='linear', x_fontsize = 34, y_fontsize = 34)
		ax3.set_xticks([.5, 1.5, 2.5])
		ax3.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(sample_sizes/N*100))
		ax3.set_yticks(np.array([g + 0.5 for g in np.arange(len(betas))]))
		ax3.set_yticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		cbar = ax3.collections[0].colorbar
		cbar.ax.tick_params(labelsize=18)
		ax3.figure.axes[-1].yaxis.label.set_size(30)
		fig3.savefig('../figures/Sampling/Networks/barabasi-albert/avg_t_aposteriori_'+model+'_p%.1f_L.pdf'%(p))
		plt.close(fig3)

		my_plot_layout(ax=ax4, xlabel=r'Sample size (%)', ylabel=r'$R_0$', yscale='linear', x_fontsize = 34, y_fontsize = 34)
		ax4.set_xticks([.5, 1.5, 2.5])
		ax4.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(sample_sizes/N*100))
		ax4.set_yticks(np.array([g + 0.5 for g in np.arange(len(betas))]))
		ax4.set_yticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		cbar = ax4.collections[0].colorbar
		cbar.ax.tick_params(labelsize=18)
		ax4.figure.axes[-1].yaxis.label.set_size(30)
		fig4.savefig('../figures/Sampling/Networks/barabasi-albert/avg_cs_aposteriori_'+model+'_p%.1f_L.pdf'%(p))
		plt.close(fig4)

		my_plot_layout(ax=ax5, xlabel=r'Sample size (%)', ylabel=r'$R_0$', yscale='linear', x_fontsize = 34, y_fontsize = 34)
		ax5.set_xticks([.5, 1.5, 2.5])
		ax5.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(sample_sizes/N*100))
		ax5.set_yticks(np.array([g + 0.5 for g in np.arange(len(betas))]))
		ax5.set_yticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		cbar = ax5.collections[0].colorbar
		cbar.ax.tick_params(labelsize=18)
		ax5.figure.axes[-1].yaxis.label.set_size(30)
		fig5.savefig('../figures/Sampling/Networks/barabasi-albert/figure_times_'+model+'_p%.1f_L.pdf'%(p))
		plt.close(fig5)

		my_plot_layout(ax=ax6, xlabel=r'Sample size (%)', ylabel=r'$R_0$', yscale='linear', x_fontsize = 34, y_fontsize = 34)
		ax6.set_xticks([.5, 1.5, 2.5])
		ax6.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(sample_sizes/N*100))
		ax6.set_yticks(np.array([g + 0.5 for g in np.arange(len(betas))]))
		ax6.set_yticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		cbar = ax6.collections[0].colorbar
		cbar.ax.tick_params(labelsize=18)
		ax6.figure.axes[-1].yaxis.label.set_size(30)
		fig6.savefig('../figures/Sampling/Networks/barabasi-albert/figure_cs_'+model+'_p%.1f_L.pdf'%(p))
		plt.close(fig6)

		i_p+=1

	        
	        
	        


