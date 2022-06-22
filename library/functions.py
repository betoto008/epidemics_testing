from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
import scipy as scipy
import scipy.integrate
import matplotlib.pyplot as plt
import math
import sys
import random
import seaborn as sns
from matplotlib.lines import Line2D

from scipy.interpolate import interp1d
from models import *


#----------------- Functions -----------------
def node_degrees(Amat):
        return Amat.sum(axis=0).reshape(N,1)
    
def my_power_law_function(x,a,b,c):
    return 0.0 + b*(x-1)**(-c)

def my_linear_function(x, a, b):
    return a+b*x

def est_function(beta, gamma):
    lambda1 = (np.sqrt(1-4*((sigma*gamma-sigma*beta)/(sigma+gamma)**2))-1)
    return (1/(lambda1))

def est_function_0(beta, gamma):
    lambda0 = (((beta)/(gamma))-1)
    return (1/(lambda0))

def cumulative_power_law(x, a, b):
    return (x**(a+1)-1)/(b**(a+1)-1)

def cumulative_power_law_2(x, a, b):
    return 1 - (x/b)**(a+1)

def my_plot_layout(ax, yscale = 'linear', xscale = 'linear', ticks_labelsize = 24,
                   xlabel = '', ylabel = '', title = '', x_fontsize=24, y_fontsize = 24,
                   t_fontsize = 24):
    ax.tick_params(labelsize = ticks_labelsize)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize = x_fontsize)
    ax.set_ylabel(ylabel, fontsize = y_fontsize)
    ax.set_title(title, fontsize = y_fontsize)

def prob_detection_stat(N, n, m):

	return (1-(scipy.special.comb(N-n,m))/(scipy.special.comb(N,m)))

def prob_detection_acum(N, n, m):

	prob_array = np.array([(1-(scipy.special.comb(N-n[0],m))/(scipy.special.comb(N,m)))])
	for i in np.arange(1,len(n)):
		prob = (1-(scipy.special.comb(N-n[i],m))/(scipy.special.comb(N,m)))
		for j in np.arange(i):
			prob = prob*((scipy.special.comb(N-n[j],m))/(scipy.special.comb(N,m)))
		prob_array =np.append(prob_array, prob)

	return prob_array

def prob_detection_acum2(N, n, m):

	prob_array = np.array([(1-(math.factorial(N-n[0])*math.factorial(N-m))/(math.factorial(N-n[0]-m)*math.factorial(N)))])
	for i in np.arange(1,len(n)):
		prob = (1-(math.factorial(N-(m*int(np.min((i,4))))-n[i])*math.factorial(N-(m*int(np.min((i,4))))-m))/(math.factorial(N-(m*int(np.min((i,4))))-n[i]-m)*math.factorial(N-(m*int(np.min((i,4)))))))
		for j in np.arange(i):
			prob = prob*((math.factorial(N-(m*int(np.min((j,4))))-n[j])*math.factorial(N-(m*int(np.min((j,4))))-m))/(math.factorial(N-(m*int(np.min((j,4))))-n[j]-m)*math.factorial(N-(m*int(np.min((j,4)))))))
		prob_array =np.append(prob_array, prob)

	return prob_array

def sort_nodes(p, beta, sigma, gamma, data_I, data_E, data_nodes, upper_limit):

	max_values = np.array([np.max(data_I[i,:]) for i in np.arange(len(data_I[:,0]))])
	#data_I = data_I[max_values!=0,:]
	#data_E = data_E[max_values!=0,:]
	#data_nodes = data_nodes[max_values!=0]
	#max_values = max_values[max_values!=0]
	#data_ext = np.array([((data_I[i,-1]==0) & (np.max(data_I[i,:]) < 20)) for i in np.arange(len(data_I[:,0]))])
	#data_ext = np.array([(((data_I[i,-1]==0) & (data_E[i,-1]==0)) & (np.max(data_I[i,:]) < upper_limit)) for i in np.arange(len(data_I[:,0]))])
	data_ext = np.array([(((data_I[i,-1]==0) & (data_E[i,-1]==0))) for i in np.arange(len(data_I[:,0]))])


	nodes_ext = data_nodes[data_ext]
	nodes_succ = data_nodes[~data_ext]
	I_ext = data_I[data_ext]
	I_epi = data_I[~data_ext]

	return nodes_ext, nodes_succ, I_ext, I_epi, max_values, data_ext

def regularize_time_series(T_total, intervals, t, E_t, I_t):

	#### Fill equally-spaced time array with samples than made it through.
	T_avg = np.linspace(0, T_total, intervals)
	E_avg_temp = np.zeros(shape = (intervals))
	I_avg_temp = np.zeros(shape = (intervals))

	if(t[-1]>=(T_total)):
		j = 0
		for k in np.arange(intervals):
			while(t[j]<T_avg[k]):
				j = j+1
			E_avg_temp[k] += E_t[j]
			I_avg_temp[k] += I_t[j]
	else:
		j=0
		k=0
		while(T_avg[k]<t[-1]):
			while(t[j]<=T_avg[k]):
				j = j+1
			E_avg_temp[k] += E_t[j]
			I_avg_temp[k] += I_t[j]
			k = k+1

	return E_avg_temp, I_avg_temp


#----------------- Plots -----------------

def plot_prob_time(T_total, sample_sizes, R0, sigma, N, func_time, func_infec, colors, log_scale = False, net_name = 'no_network', folder = ''):

	fig, ax = plt.subplots(figsize=(12,8))
	#ax.set_title(r'$R_0 = %.1f$ and $N = %.0f$'%(R0, N), fontsize = 18)
	time_x = np.linspace(0,T_total,100)
	## Run over the different sample sizes
	for m , c in zip(sample_sizes, colors):

		func_prob = prob_detection_acum2(N,func_infec.astype(int), m)
		f = interp1d(func_time, func_prob, 'quadratic')
		mean = sum(func_time*func_prob)
		std = np.sqrt(sum(np.power((func_time-mean),2)*func_prob))
		#ax.plot(func_time, func_prob,'.', c = c, ms = 10, label = r'$m = %d$ ; $\bar{t} = %.1f$ days; $\sigma_t = %.1f$ days'%(m, mean, std))
		ax.plot(time_x, f(time_x),'-', c = c, linewidth=4, label = r'$m = %d$ ; $\bar{t} = %.1f$ days; $\sigma_t = %.1f$ days'%(m, mean, std))

		#print(func_prob)

	#ax.hlines(1,0,func_time[-1], linestyle = 'dashed')
	ax.set_xlabel('Time [days]', fontsize = 30)
	ax.set_ylabel('Prob. detection', fontsize = 30)
	if(log_scale):
		ax.set_yscale('log')
	ax.tick_params(labelsize = 30)
	ax.legend(fontsize=26, loc=0)
	ax.set_xlim(0,T_total)
	ax.set_ylim(0,max(func_prob)*1.2)
	ax.set_xticks(func_time[::int(T_total/10)])
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(ax.get_xticks())
	ax2.set_xticklabels(func_infec[np.isin(func_time,ax.get_xticks())].astype(int))
	ax2.set_xlabel('Individuals', fontsize = 30)
	ax2.tick_params(labelsize = 20)
	
	#fig.savefig(folder + '/prob_detection_time_R0%.1f_sigma%.1f_N%.0f_'%(R0, sigma, N)+net_name+'.pdf')

	return fig, ax

def plot_prob_ind(I_max, sample_sizes, R0, N, func_time, func_infec, colors, log_scale = False, net_name = 'no_network', folder = ''):

	fig, ax = plt.subplots(figsize=(12,8))
	ax.set_title(r'$R_0 = %.1f$ and $N = %.0f$'%(R0, N), fontsize = 18)
	infect_x = np.linspace(1,I_max,100)
	## Run over the different sample sizes
	for m , c in zip(sample_sizes, colors):
		
		func_prob = prob_detection_acum2(N,func_infec.astype(int), m)

		f = interp1d(func_infec, func_prob, 'quadratic')
		mean = sum(func_infec*func_prob)
		std = np.sqrt(sum(np.power((func_infec-mean),2)*func_prob))
		ax.plot(func_infec, func_prob,'.', c = c, label = r'$m = %d$ ; $\bar{n} = %d$ ind; $\sigma_n = %.1f$ ind'%(m, mean, std))
		ax.plot(infect_x, f(infect_x),'-', c = c)

	#ax.hlines(1,0,func_time[-1], linestyle = 'dashed')
	ax.set_xlabel('Individuals', fontsize = 14)
	ax.set_ylabel('Prob. of detecting at least one', fontsize = 14)
	if(log_scale):
		ax.set_yscale('log')
	ax.tick_params(labelsize = 14)
	plt.legend(fontsize=15, loc=0)
	plt.xlim(1,I_max)
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(func_infec)
	ax2.set_xticklabels(['%d'%i for i in func_time])
	ax2.set_xlabel('Time [days]', fontsize = 14)
	ax2.tick_params(labelsize = 14)
	plt.ylim(0,max(func_prob)*1.1)
	plt.xlim(1,I_max)
	plt.savefig(folder + 'prob_detection_ind_R0%.1f_N%.0f_'%(R0, N)+net_name+'.pdf')

	return fig, ax

def plot_cum_prob_time(T_total, sample_sizes, R0, sigma, p, N, func_time, func_infec, func_infec2, colors, log_scale = False, net_name = 'no_network', folder = '', external_ax = False):

	ax = external_ax
	if not(external_ax):
		fig, ax = plt.subplots(figsize=(12,8))
	#ax.set_title(r'$R_0 = %.1f$ and $N = %.0f$'%(R0, N), fontsize = 18)
	time_x = np.linspace(0,T_total,200)
	## Run over the different sample sizes
	for m , c in zip(sample_sizes, colors):

		func_prob = prob_detection_acum2(N,func_infec.astype(int), m)
		#func_prob2 = prob_detection_acum2(N,func_infec2.astype(int), m)

		#f = interp1d(func_time, np.cumsum(func_prob), 'cubic')
		#f2 = interp1d(func_time, np.cumsum(func_prob2), 'cubic')
		
		np.savetxt(folder + '/prob_cum_detection_time_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(R0, sigma, N, p, m)+net_name+'.txt', (func_prob, func_prob), fmt = '%.3f')

		ax.plot(func_time, np.cumsum(func_prob), linestyle = '-', linewidth = 3, color = c, ms = 10, label = 'm = %.1f %%'%(m*100/N))
		#ax.plot(time_x, f(time_x),'-', c = c, linewidth = 3, label = 'm = %d'%m)
		#ax.plot(time_x, f2(time_x),'-', c = c, linewidth = 4, alpha = 0.5)
		#ax.vlines(time_x[np.where(f(time_x)<=0.91)][-1], 0,0.9, linestyle = 'dashed', color = c, alpha = 0.5, label = '$t_{%d} = %.1f $days'%(m, time_x[np.where(f(time_x)<=0.91)][-1]), linewidth = 3)

	ax.hlines(1,0,func_time[-1], linestyle = 'dashed', linewidth = 3)
	ax.hlines(.9,0,func_time[-1], linestyle = 'dashed', color = 'brown', label = '90%', linewidth = 3)
	#ax.hlines(1,0,func_time[-1], linestyle = 'dashed')
	ax.set_xlabel('Time [days]', fontsize = 30)
	ax.set_ylabel('Cum. Prob. detection', fontsize = 30)
	if(log_scale):
		ax.set_yscale('log')
	ax.tick_params(labelsize = 30)
	ax.legend(fontsize=20, loc=0)
	ax.set_xlim(0,T_total)
	ax.set_ylim(0,1.1)
	ax.set_xticks(func_time[::int(T_total/5)])
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(ax.get_xticks())
	ax2.set_xticklabels(func_infec[np.isin(func_time,ax.get_xticks())].astype(int))
	ax2.set_xlabel('Individuals', fontsize = 30)
	ax2.tick_params(labelsize = 20)

	#fig.savefig(folder + '/prob_cum_detection_time_R0%.1f_sigma%.1f_N%.0f_'%(R0, sigma, N)+net_name+'.pdf')
	return ax
	if not(external_ax):
		return fig, ax

def plot_cum_prob_ind(I_max, sample_sizes, R0, N, func_time, func_infec, colors, log_scale = False, net_name = 'no_network', folder = ''):

	fig, ax = plt.subplots(figsize=(12,8))
	ax.set_title(r'$R_0 = %.1f$ and $N = %.0f$'%(R0, N), fontsize = 18)
	infect_x = np.linspace(1,I_max,100)
	## Run over the different sample sizes
	for m , c in zip(sample_sizes, colors):
		
		func_prob = prob_detection_acum2(N,func_infec.astype(int), m)

		f = interp1d(func_infec, np.cumsum(func_prob), 'cubic')
		
		ax.plot(func_infec, np.cumsum(func_prob), '.', c = c, label = 'm = %d'%m)
		ax.plot(infect_x, f(infect_x),'-', c = c)
		ax.vlines(infect_x[np.where(f(infect_x)<=0.91)][-1], 0,0.9, linestyle = 'dashed', color = c, alpha = 0.5, label = '$t_{%d} > %.1f $days'%(m, infect_x[np.where(f(infect_x)<=0.91)][-1]))


	ax.hlines(1,0,func_infec[-1], linestyle = 'dashed')
	ax.hlines(.9,0,func_infec[-1], linestyle = 'dashed', color = 'brown', label = '90%')
	#ax.hlines(1,0,func_time[-1], linestyle = 'dashed')
	ax.set_xlabel('Individuals', fontsize = 14)
	ax.set_ylabel('Cum. Prob. of detecting at least one', fontsize = 14)
	if(log_scale):
		ax.set_yscale('log')
	ax.tick_params(labelsize = 14)
	plt.legend(fontsize=15, loc=0)
	plt.xlim(1,I_max)
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks(func_infec)
	ax2.set_xticklabels(['%d'%i for i in func_time])
	ax2.set_xlabel('Time [days]', fontsize = 14)
	ax2.tick_params(labelsize = 14)
	plt.xlim(1,I_max)

	plt.savefig(folder + 'prob_cum_detection_ind_R0%.1f_N%.0f_'%(R0, N)+net_name+'.pdf')

	return fig, ax

def plot_trajectory(N, G_name, beta, sigma, gamma, T_total, p, initE, initI, est, Tseries, Eseries, Iseries, I_max_1, I_max_2, time, E_solution, I_solution, folder, external_ax = False, labels = False, succ = False, plot_E=False, plot_I=False, plot_det = False):
	
	
	sns.set_style('ticks')
	sns.despine()

	lambda1 = ((-sigma-gamma)/(2)) + (1/2)*np.sqrt((sigma-gamma)**2 + 4*sigma*beta)
	est = 1/lambda1

	ax = external_ax
	if not(external_ax):
		fig, ax = plt.subplots(figsize=(12,8))

	colors_I = ['black', 'darkred', 'indigo']


	if(succ):
		if(plot_det):
			if(plot_E):
				ax.plot(Tseries, Eseries, '-', ms=10,color='darkorange', alpha = 0.6, linewidth = 4)
				ax.plot(time, E_solution, '--', ms=10, color = 'darkorange', alpha = 0.6, linewidth = 3)
			if(plot_I):
				ax.plot(Tseries, Iseries, '-', ms=10, color=colors_I[1], alpha = 0.6, linewidth = 4)
				ax.plot(time, I_solution,'--', ms=10, color = colors_I[0], alpha = 0.6, linewidth = 3)
			ax.hlines(est, 0, T_total, linestyle = 'dashed', linewidth = 4, color = 'silver', alpha = .8)
		else:
			if(plot_E):
				ax.plot(Tseries, Eseries, '-', ms=10, color='darkorange', alpha = 0.6, linewidth = 4)
			if(plot_I):
				ax.plot(Tseries, Iseries, '-', ms=10, color=colors_I[1], alpha = 0.6, linewidth = 4)
			ax.hlines(est, 0, T_total, linestyle = 'dashed', linewidth = 4, color = 'silver', alpha = .8)
	else:
		if(plot_det):
			if(plot_E):
				ax.plot(Tseries, Eseries, '-', ms=10, color='darkorange', alpha = 0.6, linewidth = 4)
				ax.plot(time, E_solution, '--', ms=10, color = 'darkorange', alpha = 0.6, linewidth = 3)
			if(plot_I):
				ax.plot(Tseries, Iseries, '-', ms=10, color=colors_I[2], alpha = 0.6, linewidth = 4)
				ax.plot(time, I_solution, '--', ms=10, color = colors_I[0], alpha = 0.6, linewidth = 3)
			ax.hlines(est, 0, T_total, linestyle = 'dashed', label = 'Establishment', linewidth = 4, color = 'silver', alpha = .8)
		else:
			if(plot_E):
				ax.plot(Tseries, Eseries, '-', ms=10,color='darkorange', alpha = 0.6, linewidth = 4)
			if(plot_I):
				ax.plot(Tseries, Iseries, '-', ms=10, color=colors_I[2], alpha = 0.6, linewidth = 4)
			ax.hlines(est, 0, T_total, linestyle = 'dashed', linewidth = 4, color = 'silver', alpha = .8)

	#ax.vlines(est*np.log(np.exp(0.577216)/(1+(1/est))), 1, est, linestyle = 'dashed', alpha = 0.3) 
	my_plot_layout(ax=ax, xlabel = 'Time [days]', ylabel = 'Individuals', yscale  ='log', x_fontsize = 34, y_fontsize = 34)
	ax.set_xlim(0,int(T_total-1))
	ax.set_ylim(0.5,I_max_2*1.1)
	lines_symbols = [Line2D([0], [0], linestyle = '-',linewidth = 4, color=colors_I[1], marker = '', ms = 12, alpha = 0.6), Line2D([0], [0], linestyle = '-',linewidth = 4, color=colors_I[2], marker = '', ms = 12, alpha = 0.6), 
	Line2D([0], [0], linestyle = '--',linewidth = 3, color=colors_I[0], marker = '', ms = 12, alpha = 0.6), Line2D([0], [0], linestyle = '--',linewidth = 4, color='silver', marker = '', ms = 12, alpha = 0.6)]
	labels_symbols = ['Established', 'Extinct', 'Deterministic', 'Establishment']
	ax.legend(lines_symbols, labels_symbols, fontsize = 24, loc = 4)

	if not(external_ax):
		fig.savefig(folder+'pdfs/trajectory_R0%.1f_N%.0f_p%.1f_'%(beta/gamma, N, p)+G_name+'.pdf')
		fig.savefig(folder+'trajectory_R0%.1f_N%.0f_p%.1f_'%(beta/gamma, N, p)+G_name+'.png', transparent=True)
		return fig, ax

def plot_ensemble(N, G_name, beta, sigma, gamma, T_total, n_ensemble, p, initI, est, T_avg, E_avg, I_avg, E_avg2, I_avg2, epi_nodes, ext_nodes, I_max_1, I_max_2, counter, time, E_solution, I_solution, folder, external_ax1 = False, external_ax2 = False, plot_E=False, plot_I=False):

	seaborn.set_style('ticks')
	seaborn.despine()

	lambda1 = ((-sigma-gamma)/(2)) + (1/2)*np.sqrt((sigma-gamma)**2 + 4*sigma*beta)
	est = 1/lambda1

	ax1 = external_ax1
	if not(external_ax1):
		fig1, ax1 = plt.subplots(figsize=(12,8)) #, gridspec_kw={'width_ratios': [2,1]}

	E_var = E_avg2 - E_avg**2
	I_var = I_avg2 - I_avg**2

	if(plot_E):
		ax1.plot(T_avg, E_avg, '.', color = 'darkorange', ms=15, label = 'Simulation E')
		ax1.plot(time, E_solution,'-', color = 'darkorange', ms=8, label='$R_0^*$ approx.', linewidth = 4)
		ax1.fill_between(T_avg, E_avg-np.sqrt(E_var), E_avg+np.sqrt(E_var), color = 'darkorange', alpha=0.4)
	if(plot_I):
		ax1.plot(T_avg, I_avg, '.', color = 'darkred', ms=15, label = 'Simulation I')
		ax1.plot(time, I_solution,'-', color = 'darkred', ms=8, label='$R_0^*$ approx.', linewidth = 4)
		ax1.fill_between(T_avg, I_avg-np.sqrt(I_var), I_avg+np.sqrt(I_var), color = 'darkred', alpha=0.4)

	ax1.hlines(est, 0, T_total, linestyle = 'dashed', label = 'Establishment', linewidth = 4)
	#ax[0].vlines(est*np.log(np.exp(0.577216)/(1+(1/est))), 1, est, linestyle = 'dashed', alpha = 0.5) 
	ax1.set_xlim(0,int(T_total-1))
	ax1.legend(fontsize=14)
	ax1.set_xlabel('Time [days]', fontsize = 30)
	ax1.set_ylabel('Indiv.', fontsize = 30)
	#ax1.set_title(r'$R_0 = %.01f$ ; $N = %.0f$ ; $p=%.1f$ ; sim with %d/%d not extinct'%(beta/gamma, N, p, counter,n_ensemble), fontsize = 18)
	ax1.tick_params(labelsize = 30)
	ax1.set_ylim(0.5,max(I_max_1*2, I_max_2*2))
	ax1.legend(fontsize = 26, loc = 2)
	ax1.set_yscale('log')

	if not(external_ax1):
		fig1.savefig(folder+'/ensemble_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.pdf')

	ax2 = external_ax2
	if not(external_ax2):
		fig2, ax2 = plt.subplots(figsize=(12,8)) #, gridspec_kw={'width_ratios': [2,1]}

	ax2.hist(ext_nodes, color = 'r', alpha = 0.5, label= 'Extint', density = True, bins = range(50))
	ax2.hist(epi_nodes, color = 'b', alpha = 0.5, label= 'Succesful', density = True, bins = range(50))
	ax2.set_xlabel('Degree', fontsize = 30)
	ax2.set_ylabel('Prob.', fontsize = 30)
	#ax2.set_title('Degree of first infected node', fontsize = 16)
	ax2.tick_params(labelsize = 30)
	ax2.legend()
	ax2.set_yscale('log')

	if not(external_ax2):
		fig2.savefig(folder+'/histograms_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.pdf')

	if not(external_ax1):
		if not(external_ax2):
			return fig1, ax1, fig2, ax2

#----------------- Models -----------------
def run_deterministic(N, beta, sigma, gamma, p, T_total, folder):

	#### Fill array with analytical solution
	lambda1 = ((-sigma-gamma)/(2)) + (1/2)*np.sqrt((sigma-gamma)**2 + 4*sigma*beta)
	lambda2 = ((-sigma-gamma)/(2)) - (1/2)*np.sqrt((sigma-gamma)**2 + 4*sigma*beta)
	#print('lambda_1 = ', lambda1)
	c1 = (((lambda1-lambda2)/beta))**(-1)
	c2 = -c1
	time = np.linspace(0, T_total, T_total+1)
	E_solution = c1*np.exp(lambda1*time) + c2*np.exp(lambda2*time)
	I_solution = c1*np.exp(lambda1*time)*((lambda1+sigma)/(beta)) + c2*np.exp(lambda2*time)*((lambda2+sigma)/(beta))
	sol_total_approx = c1*np.exp(lambda1*time)*(1+((lambda1+sigma)/(beta)))
	I_max_2 = max(np.concatenate((E_solution, I_solution)))

	np.savetxt(folder+'/deterministic_analytic_R0%.1f_sigma%.1f_p%.1f_N%d.txt'%(beta/gamma, sigma, p, N), (time, E_solution, I_solution), fmt = '%.3f')

	return lambda1, lambda2, time, E_solution, I_solution, sol_total_approx, I_max_2

def run_network_trajectory(N, G, beta, sigma, gamma, T_total, p, initE, initI, folder = '', save = False):

	## Exting (0) or succesful (1) 
	status = 0
	model = SEIRSNetworkModel(G       =G, 
	                          beta    =beta, 
	                          sigma   =sigma, 
	                          gamma   =gamma, 
	                          p = p,
	                          initE = initE,
	                          initI = initI,
	                          store_Xseries=True)
	##Run model
	model.run(T=T_total*1.1, print_interval = False)
	Eseries = model.numE
	Iseries = model.numI

	#### Get degree of initial infected node
	init_node = np.where(model.Xseries[0,:]==3)[0][0]
	init_degree = G.degree(init_node)

	## Change status
	#if(model.numI[-1]>0 and np.max(model.numI)>10):
	#    status = 1
	#if(model.numI[-1]>0 and model.numE[-1]>0):
	#    status = 1
	I_peak = N*((sigma*gamma)/(beta*(sigma+gamma)))*(beta/gamma - 1 - np.log(beta/gamma))
	print(I_peak, np.max(model.numI))
	if(np.max(model.numI)>I_peak):
		status = 1

	if (save):
		np.savetxt(folder+'/Xseries_R0%.2f_sigma%.2f_N%d_p%.1f.txt'%(beta/gamma, sigma, N, p), (model.Xseries), fmt = '%d')

	return model.tseries, model.numE, model.numI, status

def run_network_ensemble(N, G, G_name, beta, sigma, gamma, T_total, intervals, n_ensemble, p, initE, initI, folder, stochastic, sampling, sample_sizes, aposteriori = False, slope = None):


	T_avg = np.linspace(0, T_total, intervals)

	fitness = ((-sigma-gamma)/(2)) + (1/2)*np.sqrt((sigma-gamma)**2 + 4*sigma*beta)
	est = 1/(fitness)
	counter = 0

	if(stochastic):

		#Open files for trajectories

		file_E = open(folder+'/ensemble_E_R0%.2f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')
		file_I = open(folder+'/ensemble_I_R0%.2f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')
		file_stats = open(folder+'/stats_R0%.2f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')

	#Open file to save sammpling data
	if(sampling):
		file_sampling1 = open(folder+'/sampling_stats_R0%.2f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, sample_sizes[0])+G_name+'.txt', 'a')
		#file_n1 = open(folder+'/detect_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 150)+G_name+'.txt', 'a')
		file_sampling2 = open(folder+'/sampling_stats_R0%.2f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, sample_sizes[1])+G_name+'.txt', 'a')
		#file_n2 = open(folder+'/detect_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 250)+G_name+'.txt', 'a')
		file_sampling3 = open(folder+'/sampling_stats_R0%.2f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, sample_sizes[2])+G_name+'.txt', 'a')
		#file_n3 = open(folder+'/detect_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 400)+G_name+'.txt', 'a')

	#### Run ensemble of SEIR simulations
	for i in np.arange(n_ensemble):

		G = nx.barabasi_albert_graph(N, 2)
		nodeDegrees = np.array([d[1] for d in G.degree()])
		#file_network = open(folder+'/network_degree_distrib_N%d.txt'%(N),'a')
		#np.savetxt(file_network, nodeDegrees,fmt = '%d')
		#file_network.close()
		meanDegree = np.mean(nodeDegrees)
		
		model = SEIRSNetworkModel(G       =G, 
	                              beta    =beta, 
	                              sigma   =sigma, 
	                              gamma   =gamma, 
	                              p = p,
	                              initE = initE,
	                              initI = initI,
	                              store_Xseries=True)
		model.run(T=T_total*1.1, print_interval = False)
		epidemic = 0
		extinction = 0
		#### Get degree of initial infected node
		init_node = np.where(model.Xseries[0,:]==3)[0][0]
		init_degree = G.degree(init_node)

		####### IF STOCHASTIC #######
		if(stochastic):
			#create temporal arrays
			E_avg_temp = np.zeros(shape = (intervals))
			I_avg_temp = np.zeros(shape = (intervals))

			#### Fill equally-spaced time array with samples than made it through.
			if(model.tseries[-1]>=(T_total)):
				if(np.max(model.numI)>(est)):
					epidemic = 1
					counter +=1
				else:
					extinction = 1
				j = 0
				for k in np.arange(intervals):
					while(model.tseries[j]<T_avg[k]):
						j = j+1
					E_avg_temp[k] += model.numE[j]
					I_avg_temp[k] += model.numI[j]
			else:
				if(np.max(model.numI)>(est)):
					epidemic = 1
					counter +=1
				else:
					extinction = 1
				j=0
				k=0
				while(T_avg[k]<model.tseries[-1]):
					while(model.tseries[j]<=T_avg[k]):
						j = j+1
					E_avg_temp[k] += model.numE[j]
					I_avg_temp[k] += model.numI[j]
					k = k+1

			#### Save all trajectories
			np.savetxt(file_E, E_avg_temp, fmt = '%.3f', newline = ' ')
			file_E.write("\n")
			np.savetxt(file_I, I_avg_temp, fmt = '%.3f', newline = ' ')
			file_I.write("\n")
			np.savetxt(file_stats, np.array([init_degree,  epidemic, extinction]), fmt = '%d', delimiter = ' ', newline = ' ')
			file_stats.write("\n")

		####### IF SAMPLING #######
		if(sampling):
			if(model.tseries[-1]>=(T_total)):
				if(np.max(model.numI)>(est*2)):
					epidemic = 1
					counter +=1
			else:
				if(np.max(model.numI)>(est*2)):
					epidemic = 1
					counter +=1
				else:
					extinction = 1

			run_sampling_protocol_2(N, T_total, sample_sizes[0], model.tseries, model.Xseries, model.degree, epidemic, extinction, file_sampling1, posteriori = aposteriori, slope = slope)
			run_sampling_protocol_2(N, T_total, sample_sizes[1], model.tseries, model.Xseries, model.degree, epidemic, extinction, file_sampling2, posteriori = aposteriori, slope = slope)
			run_sampling_protocol_2(N, T_total, sample_sizes[2], model.tseries, model.Xseries, model.degree, epidemic, extinction, file_sampling3, posteriori = aposteriori, slope = slope)

	
	if(stochastic):
		#save stat ensemble files
		##np.savetxt(file_degree_epi, epi_nodes, fmt = '%d')
		##np.savetxt(file_degree_ext, ext_nodes, fmt = '%d')
		##np.savetxt(file_degree_no_epi, no_epi_nodes, fmt = '%d')

		#Close stat ensemble files
		#file_degree_epi.close()
		#file_degree_ext.close()
		#file_degree_no_epi.close()

		#Close trajectories files
		file_E.close()
		file_I.close()
		file_stats.close()

		

	#Close sampling files
	if(sampling):
		file_sampling1.close()
		#file_n1.close()
		file_sampling2.close()
		#file_n2.close()
		file_sampling3.close()
		#file_n3.close()

	print(counter)

	#I_max_1 = max(np.concatenate((E_avg_succ/counter,I_avg_succ/counter)))

	#### Fill array with analytical solution
	#lambda1, lambda2, time, E_solution, I_solution, sol_total_approx, I_max_2 = run_deterministic(N, beta, sigma, gamma, T_total)

	#np.savetxt(folder+'/deterministic_R0%.1f_sigma%.1f.txt'%(beta/gamma, sigma), (time, E_solution, I_solution), fmt = '%.3f')
	
	#To be changed by a function that calculate this averages
	#np.savetxt(folder+'/ensemble_avg_E_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, E_avg_succ/counter, E_avg_ext/(n_ensemble-counter), E_avg_total/n_ensemble), fmt = '%.3f')
	#np.savetxt(folder+'/ensemble_avg_I_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, I_avg_succ/counter, I_avg_ext/(n_ensemble-counter), I_avg_total/n_ensemble), fmt = '%.3f')
	#np.savetxt(folder+'/ensemble_avg_E2_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, E_avg2_succ/counter, E_avg2_ext/(n_ensemble-counter), E_avg2_total/n_ensemble), fmt = '%.3f')
	#np.savetxt(folder+'/ensemble_avg_I2_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, I_avg2_succ/counter, I_avg2_ext/(n_ensemble-counter), I_avg2_total/n_ensemble), fmt = '%.3f')

	
	
	#return T_avg, E_avg_succ/counter, I_avg_succ/counter, E_avg2_succ/counter, I_avg2_succ/counter, E_avg_ext/(n_ensemble-counter), I_avg_ext/(n_ensemble-counter), E_avg2_ext/(n_ensemble-counter), I_avg2_ext/(n_ensemble-counter), E_avg_total/n_ensemble, I_avg_total/n_ensemble, E_avg2_total/n_ensemble, I_avg2_total/n_ensemble, epi_nodes, ext_nodes, I_max_1, I_max_2, counter, time, E_solution, I_solution, sol_total_approx
	return True

def run_sampling(N, beta, sigma, gamma, T_total, p, m, I_series, folder, file_time, file_n, G_name):
	#### Samping protocol
	#for m in sample_sizes:
	#file_time = open(folder+'/detect_time_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, m)+G_name+'.txt', 'a')
	#file_n = open(folder+'/detect_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, m)+G_name+'.txt', 'a')
	time_first_detec = np.array([])
	total_n_detect = np.array([])
	##Run over trajectories
	#for i in range(len(I_matrix[:,0])):
	first_detec = False
	n_detect = 0
	## Run over time in a single trajectory
	for t in range(T_total+1):
		sample = np.array(random.sample(range(N-min(t,7)*m),m))
		n_detect_t = sum(np.isin(sample,range(int(I_series[t]))))
		if(n_detect_t):
			if not(first_detec):
				time_first_detec = np.append(time_first_detec,t)
				first_detec = True
		n_detect += n_detect_t
	total_n_detect = np.append(total_n_detect,n_detect)

	np.savetxt(file_time, time_first_detec, fmt = '%d')
	np.savetxt(file_n, total_n_detect, fmt = '%d')
	#file_time.close()
	#file_n.close()
    #####

def run_sampling_protocol(N, T_total, m, tseries, Xseries, degree, epidemic, extinction, file_sampling, posteriori = False, slope = None):

	degree = degree.reshape(1,N)[0]
	nodes = np.arange(N)
	meanDegree = np.mean(degree)
	days = np.arange(T_total+1)
	sampling_array = nodes
	#### Samping protocol
	time_first_detec = 1000
	#total_n_detect = 0
	first_n_detec = 0
	sampled_nodes = np.array([])
	first_detec = False
	n_detect = 0
	j = 0
	for t in days:
		if(t<tseries[-1]):
			#Replace after 7 days the first m sampled nodes
			if(t>4):
				sampling_array = np.concatenate((sampling_array,sampled_nodes[:m]))
				sampling_array = np.sort(sampling_array)
				sampled_nodes = sampled_nodes[m:]

			while(tseries[j]<=t):
				j+=1
			infected_nodes = np.where(Xseries[j,:]==3)[0]
			if(posteriori):
				#p_degree = np.array([max(1,(-1.3 + 1.9*np.log10(degree[int(i)]))) for i in sampling_array])
				p_degree = np.array([slope[int(degree[int(i)])] for i in sampling_array])
				p_degree = p_degree/np.sum(p_degree)
				sample = np.random.choice(a = sampling_array, size = m, replace = False, p = p_degree)
			else:
				sample = np.random.choice(a = sampling_array, size = m, replace = False)
			#sample = np.array(random.sample(list(sampling_array), m))
			n_detect_t = np.sum(np.isin(sample, infected_nodes))
			if(n_detect_t):
				if not(first_detec):
					time_first_detec = t
					first_n_detec = n_detect_t
					first_detec = True
			n_detect += n_detect_t
			temp1 = np.isin(sampling_array, sample)
			sampling_array = sampling_array[~temp1]
			sampled_nodes = np.concatenate((sampled_nodes, sample))
	#total_n_detect = np.append(total_n_detect,n_detect)

	np.savetxt(file_sampling, np.array([time_first_detec, first_n_detec, n_detect, epidemic, extinction]), fmt = '%d', delimiter = ' ', newline = ' ')
	file_sampling.write("\n")
	#np.savetxt(file_n, total_n_detect, fmt = '%d')

def run_sampling_protocol_2(N, T_total, m, tseries, Xseries, degree, epidemic, extinction, file_sampling, posteriori = False, slope = None):

	degree = degree.reshape(1,N)[0]
	nodes = np.arange(N)
	meanDegree = np.mean(degree)
	days = np.arange(T_total+1)
	sampling_array = nodes
	#### Samping protocol
	time_first_detec = 1000
	#total_n_detect = 0
	first_n_detec = 0
	sampled_nodes = np.array([])
	first_detec = False
	n_detect = 0
	j = 0
	cluster_size = 0
	for t in days:
		if(t<tseries[-1]):
			#Replace after 4 days the first m sampled nodes
			if(t>4):
				sampling_array = np.concatenate((sampling_array,sampled_nodes[:m]))
				sampling_array = np.sort(sampling_array)
				sampled_nodes = sampled_nodes[m:]

			while(tseries[j]<=t):
				j+=1
			infected_nodes = np.where(Xseries[j,:]==3)[0]
			if(posteriori):
				#p_degree = np.array([max(1,(-1.3 + 1.9*np.log10(degree[int(i)]))) for i in sampling_array])
				p_degree = np.array([slope[int(degree[int(i)])] for i in sampling_array])
				p_degree = p_degree/np.sum(p_degree)
				sample = np.random.choice(a = sampling_array, size = m, replace = False, p = p_degree)
			else:
				sample = np.random.choice(a = sampling_array, size = m, replace = False)
			#sample = np.array(random.sample(list(sampling_array), m))
			n_detect_t = np.sum(np.isin(sample, infected_nodes))
			if(n_detect_t):
				if not(first_detec):
					time_first_detec = t
					first_n_detec = n_detect_t
					cluster_size = len(infected_nodes)
					first_detec = True
			n_detect += n_detect_t
			temp1 = np.isin(sampling_array, sample)
			sampling_array = sampling_array[~temp1]
			sampled_nodes = np.concatenate((sampled_nodes, sample))
	#total_n_detect = np.append(total_n_detect,n_detect)

	np.savetxt(file_sampling, np.array([time_first_detec, first_n_detec, n_detect, epidemic, extinction, cluster_size]), fmt = '%d', delimiter = ' ', newline = ' ')
	file_sampling.write("\n")
	#np.savetxt(file_n, total_n_detect, fmt = '%d')

def run_network_tree(N, G, G_name, beta, sigma, gamma, T_total, intervals, n_ensemble, p, initE, initI, folder, stochastic, sampling, sample_sizes, aposteriori = False, slope = None):
	#G = nx.barabasi_albert_graph(N, 2)
	nodeDegrees = [d[1] for d in G.degree()]
	file_network = open(folder+'/network_degree_distrib.txt','a')
	np.savetxt(file_network, nodeDegrees,fmt = '%d')
	file_network.close()
	meanDegree = np.mean(nodeDegrees)

	T_avg = np.linspace(0, T_total, intervals)
	
	#E_avg_succ = np.zeros(shape = (intervals))
	#I_avg_succ = np.zeros(shape = (intervals))
	#E_avg2_succ = np.zeros(shape = (intervals))
	#I_avg2_succ = np.zeros(shape = (intervals))

	#E_avg_ext = np.zeros(shape = (intervals))
	#I_avg_ext = np.zeros(shape = (intervals))
	#E_avg2_ext = np.zeros(shape = (intervals))
	#I_avg2_ext = np.zeros(shape = (intervals))

	#E_avg_total = np.zeros(shape = (intervals))
	#I_avg_total = np.zeros(shape = (intervals))
	#E_avg2_total = np.zeros(shape = (intervals))
	#I_avg2_total = np.zeros(shape = (intervals))

	fitness = ((-sigma-gamma)/(2)) + (1/2)*np.sqrt((sigma-gamma)**2 + 4*sigma*beta)
	est = 1/(fitness)
	counter = 0

	if(stochastic):
		ext_nodes = np.array([])
		epi_nodes = np.array([])
		no_epi_nodes = np.array([])
		#Open files for trajectories

		file_E = open(folder+'/ensemble_E_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')
		file_I = open(folder+'/ensemble_I_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')
		file_degree_epi = open(folder+'/degree_epi_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')
		file_degree_ext = open(folder+'/degree_ext_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')
		file_degree_no_epi = open(folder+'/degree_no_epi_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', 'a')

	#Open file to save sammpling data
	if(sampling):
		file_sampling1 = open(folder+'/sampling_stats_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 150)+G_name+'.txt', 'a')
		#file_n1 = open(folder+'/detect_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 150)+G_name+'.txt', 'a')
		file_sampling2 = open(folder+'/sampling_stats_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 250)+G_name+'.txt', 'a')
		#file_n2 = open(folder+'/detect_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 250)+G_name+'.txt', 'a')
		file_sampling3 = open(folder+'/sampling_stats_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 400)+G_name+'.txt', 'a')
		#file_n3 = open(folder+'/detect_R0%.1f_sigma%.1f_N%.0f_p%.1f_m%d_'%(beta/gamma,sigma, N, p, 400)+G_name+'.txt', 'a')

	#### Run ensemble of SEIR simulations
	for i in range(n_ensemble):
		
		model = SEIRSNetworkModel(G       =G, 
	                              beta    =beta, 
	                              sigma   =sigma, 
	                              gamma   =gamma, 
	                              p = p,
	                              initE = initE,
	                              initI = initI,
	                              store_Xseries=True)
		model.run(T=T_total*1.1, print_interval = False)
		epidemic = 0
		extinction = 0
		#### Get degree of initial infected node
		init_node = np.where(model.Xseries[0,:]==3)[0][0]
		init_degree = G.degree(init_node)

		####### IF STOCHASTIC #######
		if(stochastic):
			#create temporal arrays
			E_avg_temp = np.zeros(shape = (intervals))
			I_avg_temp = np.zeros(shape = (intervals))

			#### Fill equally-spaced time array with samples than made it through.
			if(model.tseries[-1]>=(T_total)):
				if(model.numI[-1]>est):
					epidemic = 1
					counter +=1
					epi_nodes = np.append(epi_nodes, init_degree)
				else:
					counter +=1
					no_epi_nodes = np.append(no_epi_nodes, init_degree)

				j = 0
				for k in range(intervals):
					while(model.tseries[j]<T_avg[k]):
						j = j+1
					E_avg_temp[k] += model.numE[j]
					I_avg_temp[k] += model.numI[j]

				#E_avg_succ += E_avg_temp
				#I_avg_succ += I_avg_temp
				#E_avg2_succ += E_avg_temp**2
				#I_avg2_succ += I_avg_temp**2 

			else:
				extinction = 1
				ext_nodes = np.append(ext_nodes,init_degree)
				j=0
				k=0
				while(T_avg[k]<model.tseries[-1]):
					while(model.tseries[j]<=T_avg[k]):
						j = j+1
					E_avg_temp[k] += model.numE[j]
					I_avg_temp[k] += model.numI[j]
					k = k+1

				#E_avg_ext += E_avg_temp
				#I_avg_ext += I_avg_temp
				#E_avg2_ext += E_avg_temp**2
				#I_avg2_ext += I_avg_temp**2

			#### Save all trajectories
			np.savetxt(file_E, E_avg_temp, fmt = '%.3f', newline = ' ')
			file_E.write("\n")
			np.savetxt(file_I, I_avg_temp, fmt = '%.3f', newline = ' ')
			file_I.write("\n")

		####### IF SAMPLING #######
		if(sampling):
			if(model.tseries[-1]>=(T_total)):
				epidemic = 1
				counter +=1
			else:
				extinction = 1

			run_sampling_protocol(N, T_total, 150, model.tseries, model.Xseries, model.degree, epidemic, extinction, file_sampling1, posteriori = aposteriori, slope = slope)
			run_sampling_protocol(N, T_total, 250, model.tseries, model.Xseries, model.degree, epidemic, extinction, file_sampling2, posteriori = aposteriori, slope = slope)
			run_sampling_protocol(N, T_total, 400, model.tseries, model.Xseries, model.degree, epidemic, extinction, file_sampling3, posteriori = aposteriori, slope = slope)

		#E_avg_total += E_avg_temp
		#I_avg_total += I_avg_temp
		#E_avg2_total += E_avg_temp**2
		#I_avg2_total += I_avg_temp**2


	
	if(stochastic):
		#save stat ensemble files
		np.savetxt(file_degree_epi, epi_nodes, fmt = '%d')
		np.savetxt(file_degree_ext, ext_nodes, fmt = '%d')
		np.savetxt(file_degree_no_epi, no_epi_nodes, fmt = '%d')

		#Close stat ensemble files
		file_degree_epi.close()
		file_degree_ext.close()
		file_degree_no_epi.close()

		#Close trajectories files
		file_E.close()
		file_I.close()

		

	#Close sampling files
	if(sampling):
		file_sampling1.close()
		#file_n1.close()
		file_sampling2.close()
		#file_n2.close()
		file_sampling3.close()
		#file_n3.close()

	print(counter)

	#I_max_1 = max(np.concatenate((E_avg_succ/counter,I_avg_succ/counter)))

	#### Fill array with analytical solution
	#lambda1, lambda2, time, E_solution, I_solution, sol_total_approx, I_max_2 = run_deterministic(N, beta, sigma, gamma, T_total)

	#np.savetxt(folder+'/deterministic_R0%.1f_sigma%.1f.txt'%(beta/gamma, sigma), (time, E_solution, I_solution), fmt = '%.3f')
	
	#To be changed by a function that calculate this averages
	#np.savetxt(folder+'/ensemble_avg_E_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, E_avg_succ/counter, E_avg_ext/(n_ensemble-counter), E_avg_total/n_ensemble), fmt = '%.3f')
	#np.savetxt(folder+'/ensemble_avg_I_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, I_avg_succ/counter, I_avg_ext/(n_ensemble-counter), I_avg_total/n_ensemble), fmt = '%.3f')
	#np.savetxt(folder+'/ensemble_avg_E2_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, E_avg2_succ/counter, E_avg2_ext/(n_ensemble-counter), E_avg2_total/n_ensemble), fmt = '%.3f')
	#np.savetxt(folder+'/ensemble_avg_I2_R0%.1f_sigma%.1f_N%.0f_p%.1f_'%(beta/gamma, sigma, N, p)+G_name+'.txt', (T_avg, I_avg2_succ/counter, I_avg2_ext/(n_ensemble-counter), I_avg2_total/n_ensemble), fmt = '%.3f')

	
	
	#return T_avg, E_avg_succ/counter, I_avg_succ/counter, E_avg2_succ/counter, I_avg2_succ/counter, E_avg_ext/(n_ensemble-counter), I_avg_ext/(n_ensemble-counter), E_avg2_ext/(n_ensemble-counter), I_avg2_ext/(n_ensemble-counter), E_avg_total/n_ensemble, I_avg_total/n_ensemble, E_avg2_total/n_ensemble, I_avg2_total/n_ensemble, epi_nodes, ext_nodes, I_max_1, I_max_2, counter, time, E_solution, I_solution, sol_total_approx
	return True

