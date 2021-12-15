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
k_array = np.logspace(np.log10(2), np.log10(150), 50)
infile_k.close()
infile_p_k = open(Text_files_path+'barabasi-albert/p_k.pck','rb')
p_k = pickle.load(infile_p_k)
infile_p_k.close()

meanDegree = np.sum(k*p_k)
meanDegree2 = np.sum(k**2*p_k)
T_c = meanDegree/(meanDegree2-meanDegree)


ps=[0.0, 1.0]
gamma = 1/6
sigmas=[1000, 1/4]
models = ['SIR', 'SEIR']
R0sp1 = np.flip(np.array([1.2, 2.0, 3.0, 4.5, 6.0]))
R0sp0 = np.flip(np.array([0.8, 1.2, 2.0, 3.0, 4.5]))
R0S = np.array([R0sp0,R0sp1], dtype="object")
tau_SIR = 1/gamma
tau_SEIR = 2*(1/4+gamma)**(-1)
u = np.linspace(0.00005,0.9,100000)
colors_R = plt.cm.Paired(np.arange(6))


for q, p in enumerate(ps):
	R0s = R0S[q]
	betas= R0s*gamma
	
	for s, sigma in enumerate(tqdm(sigmas)):
		model = models[s]
		if(sigma==1000): #SIR
			Ts = 1-np.array([np.sum(p_k*(1/(1+(R/(k**2))))) for R in R0s])
			u_sols = np.array([u[np.array([np.sum(p_k*k*(1+(i-1)*j)**(k-1)) for i in u])>(np.sum(p_k*k)*u)][-1] for j in Ts])
			S = 1 - np.array([np.sum(p_k*(1-j+(i*j))**k) for (i, j) in zip(u_sols, Ts)])
		if(sigma==1/4): #SEIR
			R0s = np.array([np.sqrt(1-4*((sigma*gamma-sigma*b)/(sigma+gamma)**2)) for b in betas])
			Ts = 1-np.array([np.sum(p_k*(1/(1+(R/(k**2))))) for R in R0s])
			#Ts = 1-np.array([np.sum(p_k*(1/(1+((b*tau_SEIR)/(k*(k)))))) for b in betas])
			u_sols = np.array([u[np.array([np.sum(p_k*k*(1+(i-1)*j)**(k-1)) for i in u])>(np.sum(p_k*k)*u)][-1] for j in Ts])
			S = 1 - np.array([np.sum(p_k*(1-j+(i*j))**(k))*(np.sum(k*p_k*(1-j+(i*j))**(k))/(meanDegree)) for (i, j) in zip(u_sols, Ts)])
			#S = 1 - np.array([np.sum(p_k*(1-j+(i*j))**(2*k)) for (i, j) in zip(u_sols, Ts)])


		fig, ax = plt.subplots(figsize = (10,8), gridspec_kw={'bottom': 0.13,'left':.2})
		for b, beta in enumerate(betas):
			u_sol = u_sols[b]
			T = Ts[b]
			if(sigma==1000):
				e_k = 1-(1-T+(T*u_sol))**k_array
			if(sigma==1/4):
				e_k = 1-(1-T+(T*u_sol))**(k_array)*(1-T+(T*u_sol))**(k_array)
				#e_k = 1-((1-T+(T*u_sol))**(k)*(np.sum(k*p_k*(1-T+u_sol*T)**(k-1)))/(meanDegree))

			Log_Likelihood = np.log10(e_k/S[b])
			ax.plot(k_array/meanDegree, Log_Likelihood, color = colors_R[b], linestyle = '--', linewidth = 3, label = r'%.1f'%(R0s[b]))

			#----METHOD 1----
			#----Load data with simulation outcomes----
			data_stats = np.loadtxt(Text_files_path + 'barabasi-albert/k_normalization/stats_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.txt'%(beta/gamma, sigma, N, p))
			#data_I = np.loadtxt('../../../../Dropbox/Research/Epidemiology_2020/Text_files/Stochastic/Networks/barabasi-albert/k_normalization/ensemble_I_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.txt'%(beta/gamma, sigma, N, p))
			#data_E = np.loadtxt('../../../../Dropbox/Research/Epidemiology_2020/Text_files/Stochastic/Networks/barabasi-albert/k_normalization/ensemble_E_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.txt'%(beta/gamma, sigma, N, p))
			#max_values = np.array([np.max(data_I[i,:]) for i in np.arange(len(data_I[:,0]))])
			#data_ext = np.array([(((data_I[i,-1]==0) and (data_E[i,-1]==0)) and (np.max(data_I[i,:]) < 20)) for i in np.arange(len(data_I[:,0]))])
			
			infile_analized_data = open(Text_files_path+'barabasi-albert/k_normalization/analized_data_R0%.1f_sigma%.1f_N%d_p%.1f_barabasi-albert.pck'%(beta/gamma, sigma, N, p), 'rb')
			nodes_ext , nodes_epi, I_ext, I_epi, max_values, data_ext =  pickle.load(infile_analized_data)
			nodes_epi = data_stats[:,0][~data_ext]
			nodes_ext = data_stats[:,0][data_ext]

			z_succ_nodes = len(nodes_epi)
			z_ext_nodes = len(nodes_ext)
			total = len(nodes_epi)+len(nodes_ext)

			### Calculate histograms
			data = np.histogram(data_stats[:,0], bins=np.logspace(np.log10(1), np.log10(np.max(data_stats[:,0])), 12), density = False)
			data_cum = np.cumsum(data[0])/total
			data_succ = np.histogram(np.concatenate((nodes_epi,[])), bins=np.logspace(np.log10(1), np.log10(np.max(data_stats[:,0])), 12), density = False)
			data_succ_cum = np.cumsum(data_succ[0])/z_succ_nodes
			#print(data_cum[-1], data_succ_cum[-1])
			Log_Likelihood_data = np.log10(np.gradient(data_succ_cum)/np.gradient(data_cum))[~np.isnan(np.log10(np.gradient(data_succ_cum)/np.gradient(data_cum)))]
			degrees_data = ((data[1][1:]+data[1][:-1])/2)[~np.isnan(np.log10(np.gradient(data_succ_cum)/np.gradient(data_cum)))]
			degrees_data = degrees_data[~np.isinf(Log_Likelihood_data)]
			Log_Likelihood_data = Log_Likelihood_data[~np.isinf(Log_Likelihood_data)]

			#----METHOD 2----
			degrees_epi, counts_epi = np.unique(nodes_epi, return_counts=True)
			degrees_ext, counts_ext = np.unique(nodes_ext, return_counts=True)

			p_epi = np.sum(counts_epi)/(np.sum(counts_epi)+np.sum(counts_ext))
			max_degree = np.max(np.concatenate((degrees_epi, degrees_ext)))
			degrees = np.array(np.arange(1,int(max_degree)))
			prob_ext_k0_data2 = np.array([])
			n_epi = 0
			n_ext = 0

			for d in degrees[:]:
				if(counts_ext[degrees_ext==d].size>0 and counts_epi[degrees_epi==d].size>0):
					n_epi_d = counts_epi[degrees_epi==d][0]
					#print(n_epi)
					n_ext_d = counts_ext[degrees_ext==d][0]
					n_total_d = n_epi_d + n_ext_d
					prob_ext_k0_data2 = np.append(prob_ext_k0_data2, (n_ext_d)/n_total_d)
				else:
					degrees = degrees[~np.isin(degrees, d)]

			Log_Likelihood_data2 = np.log10((1-prob_ext_k0_data2)/p_epi)
			#degrees_array = np.linspace(np.min(degrees_data), np.max(degrees_data), 200)

			#ax.plot(degrees, Log_Likelihood_data2, linestyle = '', marker = '^', color=colors_R[b], ms = 10)
			#ax.plot(degrees_data, Log_Likelihood_data,linestyle = '', marker = '*', color=colors_R[b], ms = 14, label = r'%.1f'%(R0s[b]))
			
			#ax.plot(k, (slope*meanDegree)*np.log10(k)-(slope*meanDegree)*np.log10(3), color = colors_R[b], linestyle = '--', linewidth = 2)
		ax.hlines(0, 0, 80, linestyle = '--', color = 'silver')
		my_plot_layout(ax = ax, yscale='linear', xscale='log', ylabel=r'$\log{\left(\frac{P(k|\mathrm{epi})}{P(k)}\right)}$', xlabel=r'$k_0/\left\langle k \right\rangle$', x_fontsize = 34, y_fontsize = 34)

		ax.legend(fontsize = 22, title = r'$R_0$', title_fontsize = 24)
		ax.set_ylim(-.2, .8)
		ax.set_xlim(2/meanDegree, 150/meanDegree)
		fig.savefig('../figures/Stochastic/Networks/barabasi-albert/log-likelihood_'+model+'_p%.1f.png'%( p))
		fig.savefig('../figures/Stochastic/Networks/barabasi-albert/pdfs/log-likelihood_'+model+'_p%.1f.pdf'%( p))
	    

