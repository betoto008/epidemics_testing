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

n50_0 = np.tile(np.array([5, 3, 2, 1, 1]), (3,1))
n50_1 = np.tile(np.array([4, 2, 1, 1]), (3,1))


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
		fig, ax = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.13, 'left': 0.14})
		fig2, ax2 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.13, 'left': 0.14})
		fig3, ax3 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.13, 'left': 0.14})
		fig4, ax4 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.13, 'left': 0.14})
		fig5, ax5 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.13, 'left': 0.14})
		fig6, ax6 = plt.subplots(figsize=(12,8), gridspec_kw={'bottom': 0.13, 'left': 0.14})

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


				violin_parts = ax.violinplot(data_sampling_uniform_cs, positions = [3*i_b + i_m ], showextrema = False, showmeans = True)

				for pc in violin_parts['bodies']:
					pc.set_facecolor(colors_m[i_m-1])
					pc.set_edgecolor(colors_m[i_m-1])
				violin_parts['cmeans'].set_color(colors_m[i_m-1])

				violin_parts2 = ax2.violinplot(data_sampling_uniform_times, positions = [3*i_b + i_m ], showextrema = False, showmeans = True)

				for pc in violin_parts2['bodies']:
					pc.set_facecolor(colors_m[i_m-1])
					pc.set_edgecolor(colors_m[i_m-1])
				violin_parts2['cmeans'].set_color(colors_m[i_m-1])
				
				if(m==sample_sizes[0]):
					violin_parts = ax3.violinplot(data_sampling_uniform_cs, positions = [2*i_b + 1 ], showextrema = False, showmeans = True)
					for pc in violin_parts['bodies']:
						pc.set_facecolor(colors_m[i_m-1])
						pc.set_edgecolor(colors_m[i_m-1])
					violin_parts['cmeans'].set_color(colors_m[i_m-1])

					violin_parts = ax3.violinplot(data_sampling_aposteriori_cs, positions = [2*i_b + 2 ], showextrema = False, showmeans = True)
					for pc in violin_parts['bodies']:
						pc.set_facecolor(colors[i_m-1])
						pc.set_edgecolor(colors[i_m-1])
					violin_parts['cmeans'].set_color(colors[i_m-1])

				if(m==sample_sizes[1]):
					violin_parts = ax4.violinplot(data_sampling_uniform_cs, positions = [2*i_b + 1 ], showextrema = False, showmeans = True)
					for pc in violin_parts['bodies']:
						pc.set_facecolor(colors_m[i_m-1])
						pc.set_edgecolor(colors_m[i_m-1])
					violin_parts['cmeans'].set_color(colors_m[i_m-1])

					violin_parts = ax4.violinplot(data_sampling_aposteriori_cs, positions = [2*i_b + 2 ], showextrema = False, showmeans = True)
					for pc in violin_parts['bodies']:
						pc.set_facecolor(colors[i_m-1])
						pc.set_edgecolor(colors[i_m-1])
					violin_parts['cmeans'].set_color(colors[i_m-1])
				if(m==sample_sizes[2]):
					violin_parts = ax5.violinplot(data_sampling_uniform_cs, positions = [2*i_b + 1 ], showextrema = False, showmeans = True)
					for pc in violin_parts['bodies']:
						pc.set_facecolor(colors_m[i_m-1])
						pc.set_edgecolor(colors_m[i_m-1])
					violin_parts['cmeans'].set_color(colors_m[i_m-1])

					violin_parts = ax5.violinplot(data_sampling_aposteriori_cs, positions = [2*i_b + 2 ], showextrema = False, showmeans = True)
					for pc in violin_parts['bodies']:
						pc.set_facecolor(colors[i_m-1])
						pc.set_edgecolor(colors[i_m-1])
					violin_parts['cmeans'].set_color(colors[i_m-1])

				i_m+=1
			i_b += 1

		ax.vlines(np.array([i*3+0.5 for i in np.arange(len(R0s)+1)]), 0, ax.get_ylim()[1]*0.9, linestyle = '--', color = 'silver')
		ax.hlines(n50[0,:], np.arange(len(R0s))*3 + 0.55, np.arange(1, len(R0s)+1)*3 + 0.45, color = 'black', lw = 2)
		lines_symbols = [Line2D([0], [0], color=colors_m[0], linestyle='', marker = 's', ms = 8), Line2D([0], [0], color=colors_m[1], linestyle='', marker = 's', ms = 8), Line2D([0], [0], color=colors_m[2], linestyle='', marker = 's', ms = 8), Line2D([0], [0], color='black', linestyle='-', lw = 2, marker = '', ms = 8)]
		labels_symbols = np.concatenate(([FormatStrFormatter('%.1f').format_ticks(np.array([sample_sizes[i]])/N*100)[0] + '%' for i in np.arange(3)] ,[r'$\eta^*$']))
		my_plot_layout(ax=ax, ylabel=r'Cluster size', xlabel=r'$R_0$', yscale='log', x_fontsize = 34, y_fontsize = 34)
		ax.set_xticks(np.arange(2,len(R0s)*3 + 1, 3))
		ax.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		ax.legend(lines_symbols, labels_symbols, fontsize = 24, title = r'$m$', title_fontsize = 24)
		fig.savefig('../figures/Sampling/Networks/barabasi-albert/avg_cs_uniform_'+model+'_p%.1f_violin_L.pdf'%(p))
		plt.close(fig)

		ax2.vlines(np.array([i*3+0.5 for i in np.arange(len(R0s)+1)]), 0, ax2.get_ylim()[1]*0.9, linestyle = '--', color = 'silver')
		lines_symbols = [Line2D([0], [0], color=colors_m[0], linestyle='', marker = 's', ms = 8), Line2D([0], [0], color=colors_m[1], linestyle='', marker = 's', ms = 8), Line2D([0], [0], color=colors_m[2], linestyle='', marker = 's', ms = 8), Line2D([0], [0], color='black', linestyle='-', marker = '', ms = 8)]
		labels_symbols = np.concatenate(([FormatStrFormatter('%.1f').format_ticks(np.array([sample_sizes[i]])/N*100)[0] + '%' for i in np.arange(3)],[r'$\eta^*$']))
		my_plot_layout(ax=ax2, ylabel=r'Cluster size', xlabel=r'$R_0$', yscale='log', x_fontsize = 34, y_fontsize = 34)
		ax2.set_xticks(np.arange(2,len(R0s)*3 + 1, 3))
		ax2.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		ax2.legend(lines_symbols, labels_symbols, fontsize = 24)
		fig2.savefig('../figures/Sampling/Networks/barabasi-albert/avg_t_uniform_'+model+'_p%.1f_violin_L.pdf'%(p))
		plt.close(fig2)

		ax3.vlines(np.array([i*2+0.5 for i in np.arange(len(R0s)+1)]), 0, ax3.get_ylim()[1]*0.9, linestyle = '--', color = 'silver')
		ax3.hlines(n50[0,:], np.arange(len(R0s))*2 + 0.55, np.arange(1, len(R0s)+1)*2 + 0.45, color = 'black', lw = 2)
		lines_symbols = [Line2D([0], [0], color=colors_m[0], linestyle='', marker = 's', ms = 10, alpha = .6), Line2D([0], [0], color=colors[0], linestyle='', marker = 's', ms = 10, alpha = .6)]
		labels_symbols = ['Uniform', 'Degree-based']
		my_plot_layout(ax=ax3, ylabel=r'Cluster size', xlabel=r'$R_0$', yscale='log', x_fontsize = 34, y_fontsize = 34)
		ax3.set_xticks(np.arange(1,len(R0s)*2 + 1, 2)+0.5)
		ax3.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		ax3.legend(lines_symbols, labels_symbols, fontsize = 24)
		fig3.savefig('../figures/Sampling/Networks/barabasi-albert/avg_cs_m1_'+model+'_p%.1f_violin_L.pdf'%(p))
		plt.close(fig3)

		ax4.vlines(np.array([i*2+0.5 for i in np.arange(len(R0s)+1)]), 0, ax4.get_ylim()[1]*0.9, linestyle = '--', color = 'silver')
		ax4.hlines(n50[0,:], np.arange(len(R0s))*2 + 0.55, np.arange(1, len(R0s)+1)*2 + 0.45, color = 'black', lw = 2)
		lines_symbols = [Line2D([0], [0], color=colors_m[1], linestyle='', marker = 's', ms = 10, alpha = .6), Line2D([0], [0], color=colors[1], linestyle='', marker = 's', ms = 10, alpha = .6)]
		labels_symbols = ['Uniform', 'Degree-based']
		my_plot_layout(ax=ax4, ylabel=r'Cluster size', xlabel=r'$R_0$', yscale='log', x_fontsize = 34, y_fontsize = 34)
		ax4.set_xticks(np.arange(1,len(R0s)*2 + 1, 2)+0.5)
		ax4.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		ax4.legend(lines_symbols, labels_symbols, fontsize = 24)
		fig4.savefig('../figures/Sampling/Networks/barabasi-albert/avg_cs_m2_'+model+'_p%.1f_violin_L.pdf'%(p))
		plt.close(fig4)

		ax5.vlines(np.array([i*2+0.5 for i in np.arange(len(R0s)+1)]), 0, ax5.get_ylim()[1]*0.9, linestyle = '--', color = 'silver')
		ax5.hlines(n50[0,:], np.arange(len(R0s))*2 + 0.55, np.arange(1, len(R0s)+1)*2 + 0.45, color = 'black', lw = 2)
		lines_symbols = [Line2D([0], [0], color=colors_m[2], linestyle='', marker = 's', ms = 10, alpha = .6), Line2D([0], [0], color=colors[2], linestyle='', marker = 's', ms = 10, alpha = .6)]
		labels_symbols = ['Uniform', 'Degree-based']
		my_plot_layout(ax=ax5, ylabel=r'Cluster size', xlabel=r'$R_0$', yscale='log', x_fontsize = 34, y_fontsize = 34)
		ax5.set_xticks(np.arange(1,len(R0s)*2 + 1, 2)+0.5)
		ax5.set_xticklabels(FormatStrFormatter('%.1f').format_ticks(R0s))
		ax5.legend(lines_symbols, labels_symbols, fontsize = 24)
		fig5.savefig('../figures/Sampling/Networks/barabasi-albert/avg_cs_m3_'+model+'_p%.1f_violin_L.pdf'%(p))
		plt.close(fig5)

		i_p+=1

	        
	        
	        


