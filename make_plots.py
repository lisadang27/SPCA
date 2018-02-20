
import numpy as np
import matplotlib.pyplot as plt


def plot_detec_syst(time, data, init):
	plt.figure(figsize=(10,3))
	plt.plot(time, data, '+', label='data')
	plt.plot(time, init, '+', label='guess')
	plt.title('Initial Guess')
	plt.xlabel('Time (BMJD)')
	plt.ylabel('Relative Flux')	
	return