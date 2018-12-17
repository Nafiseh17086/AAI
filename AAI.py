import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from novainstrumentation import smooth
import seaborn as sns
import os
import os
import numpy as np
import h5py
import ast
import seaborn
import matplotlib.pyplot as plt
from scipy.io import loadmat

def Acc_g(ACC, calibration):
    cmin = np.min(calibration)

    cmax = np.max(calibration)
    acc_g = (((ACC - cmin)/(cmax- cmin))*2)-1

    return acc_g


ACC =np.loadtxt("/home/nafiseh/Documents/code of ACC/miguel_estacao40_direito_futura_000780FC5723_2018-07-09_14-27-49.txt")[:, -3:]
calibration = np.loadtxt("/home/nafiseh/Documents/code of ACC/xyz_cal.txt")[:, -3:]
#fileM = ["/home/nafiseh/Documents/code of ACC/miguel_estacao40_direito_futura_000780FC5723_2018-07-09_14-27-49.txt"]

ACC_x = ACC[:,-1]
ACC_y = ACC[:,-2]
ACC_z = ACC[:,-3]

calibration_x = calibration[:,-1]
calibration_y = calibration[:,-2]
calibration_z = calibration[:,-3]

accg_x = Acc_g(ACC_x, calibration_x)
accg_y = Acc_g(ACC_y, calibration_y)
accg_z = Acc_g(ACC_z, calibration_z)

N = len(accg_x)

sp_x = (2/N)*abs(np.fft.fft(accg_x[1:])[:N//2])
sp_y = (2/N)*abs(np.fft.fft(accg_y[1:])[:N//2])
sp_z = (2/N)*abs(np.fft.fft(accg_z[1:])[:N//2])
# fig, axs = plt.subplots(3,1)
# axs[0].plot(sp_x)
# axs[0].set_xlim(0,100)
# axs[1].plot(sp_y)
# axs[1].set_xlim(0,100)
# axs[2].plot(sp_z)
# axs[2].set_xlim(0,100)
# plt.show()
# plt.plot(sp_x, sp_y, sp_z)
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
#                                        Read File
# ----------------------------------------------------------------------------------------------------------------------

def openMat(filename, filedir):

	#open mat should have acces to info file where fs and channel is identified
	for file in os.listdir(filedir):

		print(file)
		if(file == "Attributes.txt"):
			inf = open(filedir + "/" + file, 'r')
			AttDic = ast.literal_eval(inf.read())
			inf.close()

			fs = AttDic["fs"]
			index = AttDic["index"]
			n_clusters = AttDic["clusters"]
			mat = loadmat(filename)

			ECG_data = mat["val"][index][:]

		elif (file == "Noise2790.txt"):
			Noise = np.loadtxt(filedir + "/" + file)

	return ECG_data, fs, Noise, n_clusters



def openH5(filename, filedir):
	print(filename)

	# f = h5py.File(filename, 'r')
	h5f = h5py.File(filename, 'r')
	ECG_data = h5f['dataset_1'][:, 3]

	# ECG_Macs = [key for key in f.keys()][0]
	#
	# ECG_data_group = f[ECG_Macs + "/raw"]
	#
	# fs = f[ECG_Macs].attrs["sampling rate"] * 1.0
	#
	# for file in os.listdir(filedir):
	# 	if (file == "Noise2790.txt"):
	# 		Noise = np.loadtxt(filedir + "/" + file)
	# 	elif(file == "Attributes.txt"):
	# 		inf = open(filedir + "/" + file, 'r')
	# 		AttDic = ast.literal_eval(inf.read())
	# 		inf.close()
	# 		fs = AttDic["fs"]
	# 		n_clusters = AttDic["clusters"]
	#
	# if ("JakeHeart" in filename or "Lucas" in filename or "Tiago" in filename):
	# 	ECG_data = ECG_data_group["channel_1"][:120000, 0]
	# else:
	# 	ECG_data = ECG_data_group["channel_1"][:, 0]


	return ECG_data, fs, Noise, n_clusters

def opentxt(filename, filedir):

	if("SN" in filename):
		for file in os.listdir(filedir):
			if(file == "Attributes.txt"):
				inf = open(filedir + "/" + file, 'r')
				AttDic = ast.literal_eval(inf.read())
				inf.close()

				fs = AttDic["fs"]
				n_clusters = AttDic["clusters"]

				ECG_data = np.loadtxt(filename)

			elif(file == "Noise2790.txt"):
				Noise = np.loadtxt(filedir + "/" + file)

		return ECG_data, fs, Noise, n_clusters

	else:
		HeadDic = read_header(filename)
		fs = HeadDic["sampling rate"]
		channel = HeadDic["channels"][HeadDic["sensor"] == "ECG"] + 1

		ECG_data = np.loadtxt(filename)
		ECG_data = ECG_data[:, channel]

		for file in os.listdir(filedir):
			if(file == "Noise2790.txt"):
				Noise = np.loadtxt(filedir + "/" + file)
			elif(file == "Attributes.txt"):
				inf = open(filedir + "/" + file, 'r')
				AttDic = ast.literal_eval(inf.read())
				inf.close()

				fs = AttDic["fs"]
				n_clusters = AttDic["clusters"]

		return ECG_data, fs, Noise, n_clusters

def read_header(source_file, print_header = False):

	f = open(source_file, 'r')

	f_head = [f.readline() for i in range(3)]

	#convert to diccionary (considering 1 device only)
	head_dic = ast.literal_eval(f_head[1][24:-2])

	return head_dic

MainDir = os.path.realpath("Signals2")
Folders = os.listdir(MainDir)

for eachFolder in Folders:
	#if("Synthetic" in eachFolder):
	#print(eachFolder)
	folderPath = MainDir + "/" + eachFolder
	Files = os.listdir(folderPath)

	for eachFile in Files:
		#print(eachFile[-4:])
		print(eachFile)
		if "." not in str(eachFile):
			print("Could not open because it is not a file...")
		elif ("Attributes" in eachFile or "Noise" in eachFile):
			print("not opening " + eachFile)
		else:
			FileDir = os.path.realpath(folderPath + "/" + eachFile)
			if(str(eachFile[-4:]) == ".mat"):
				data, fs, Noise, n_clusters = openMat(FileDir, folderPath)
			elif(eachFile[-4:] == ".txt"):
				data, fs, Noise, n_clusters = opentxt(FileDir, folderPath)
			elif(eachFile[-3:] == ".h5"):
				data, fs, Noise, n_clusters = openH5(FileDir, folderPath)
			else:
				print("file in incorrect format. Tolerate formats are: .txt, .h5, .mat")

			time = np.linspace(0, len(data)/fs, len(data))
			# plt.plot(time, data/max(data))
			# plt.plot(time, Noise, 'r-o')
			# plt.xticks(np.linspace(0, len(data)/fs, 4*len(data)/fs))
			# plt.show()
			#############
			# savepath = folderPath
			# if(len(np.shape(Noise)) > 1):
			# 	for i in range(0, np.shape(Noise)[1]):
			# 		ReportClustering(data, Noise[:, i], fs, time, eachFile[:-4]+str(i), folderPath, clusters=n_clusters)
			# 		# ReportSTD(data, Noise[:, i], fs, time, savepath, eachFile[:-4]+str(i))
			# else:
			# 	# plt.plot(data)
			# 	# plt.show()
			# 	ReportClustering(data, Noise, fs, time, eachFile[:-4], folderPath, clusters=n_clusters)
			# 	# ReportSTD(data, Noise, fs, time, savepath, eachFile[:-4])
# ----------------------------------------------------------------------------------------------------------------------
#                                        Extract Features
# ----------------------------------------------------------------------------------------------------------------------
#WindMethod
def WindowStat(inputSignal, statTool, fs, window_len=50, window='hanning'):

	output = np.zeros(len(inputSignal))
	win = eval('np.' + window + '(window_len)')

	if inputSignal.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if inputSignal.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len < 3:
		return inputSignal

	inputSignal = inputSignal - np.mean(inputSignal)

	WinRange = int(window_len/2)

	sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal)-WinRange:-1]]

	# windowing
	if(statTool is 'stn'):
		WinSize = window_len
		numSeg = int(len(inputSignal) / WinSize)
		SigTemp = np.zeros(numSeg)
		for i in range(1, numSeg):
			signal = inputSignal[(i - 1) * WinSize:i * WinSize]
			SigTemp[i] = signaltonoise(signal)
		output = np.interp(np.linspace(0, len(SigTemp), len(output)), np.linspace(0, len(SigTemp), len(SigTemp)), SigTemp)
	elif(statTool is 'zcr'):
		# inputSignal = inputSignal - smooth(inputSignal, window_len=fs*4)
		# inputSignal = inputSignal - smooth(inputSignal, window_len=int(fs/10))
		# sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]]

		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - int(WinRange)] = ZeroCrossingRate(sig[i - WinRange:WinRange + i]*win)
		output = smooth(output, window_len=1024)
	elif(statTool is 'std'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.std(sig[i - WinRange:WinRange + i]*win)
	elif(statTool is 'subPks'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pks = [0]
			win_len = window_len
			while(len(pks) < 10):
				pks = detect_peaks(sig[i - int(win_len / 2):int(win_len / 2) + i], valley=False, mph=np.std(sig[i - int(win_len / 2):int(win_len / 2)+ i]))
				if(len(pks) < 10):
					win_len += int(win_len/5)
			sub_zero = pks[1] - pks[0]
			sub_end = pks[-1] - pks[-2]
			subPks = np.r_[sub_zero, (pks[1:-1] - pks[0:-2]), sub_end]
			win = eval('np.' + window + '(len(subPks))')
			output[i - int(WinRange)] = np.mean(subPks*win)
	elif (statTool is 'findPks'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pks = detect_peaks(sig[i - WinRange:WinRange + i], valley=False,
								   mph=np.std(sig[i - WinRange:WinRange + i]))
			LenPks = len(pks)
			output[i - int(WinRange)] = LenPks
	elif(statTool is 'sum'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.sum(abs(sig[i - WinRange:WinRange + i] * win))
	elif(statTool is 'AmpDiff'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			win_len = window_len
			tempSig = sig[i - int(win_len / 2):int(win_len / 2) + i]
			maxPks = detect_peaks(tempSig, valley=False,
								   mph=np.std(tempSig))
			minPks = detect_peaks(tempSig, valley=True,
								   mph=np.std(tempSig))
			AmpDiff = np.sum(tempSig[maxPks]) - np.sum(tempSig[minPks])
			output[i - WinRange] = AmpDiff
	elif(statTool is 'MF'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			f, Pxx = PowerSpectrum(inputSignal[i - WinRange:i + WinRange], fs=fs, nperseg=WinRange/2)
			mf = MF_calculus(Pxx)
			output[i - WinRange] = mf
	elif(statTool is "SumPS"):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			f, Pxx = PowerSpectrum(inputSignal[i - WinRange:i + WinRange], fs=fs, nperseg=WinRange / 2)
			sps = SumPowerSpectrum(Pxx)
			output[i - WinRange] = sps
	elif(statTool is 'fractal'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = entropy(sig[i - WinRange:WinRange + i]*win)
			output[np.where(output is "nan" or output > 1E308)[0]] = 0
	elif(statTool is "AmpMean"):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.mean(abs(sig[i - WinRange:WinRange + i]) * win)
	elif(statTool is"Spikes1"):
		ss = 0.1*max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(sig[i - WinRange:WinRange + i] * win, mph=ss)
			output[i - WinRange] = pkd
	elif (statTool is "Spikes2"):
		ss = 0.1 * max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(sig[i - WinRange:WinRange + i] * win, mph=ss)
			output[i - WinRange] = md
	elif (statTool is "Spikes3"):
		ss = 0.1 * max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(abs(sig[i - WinRange:WinRange + i] )* win, mph=ss)
			output[i - WinRange] = md

	output = output - np.mean(output)
	output = output/max(output)
	#output = smooth(output, window_len=10)

	return output
print("Extracting features...")

# 1 - Std Window
signalSTD = WindowStat(sp_x , fs=fs, statTool='std', window_len=(win * fs) / 256)

print("...feature 1 - STD")
#
# # 2 - ZCR
# signalZCR64 = WindowStat(signal, fs=fs, statTool='zcr', window_len=(win * fs) / 512)
#
# print("...feature 2 - ZCR")
# # 3 - Sum
# signalSum64 = WindowStat(signal, fs=fs, statTool='sum', window_len=(win * fs) / 256)
# signalSum128 = WindowStat(signal, fs=fs, statTool='sum', window_len=(win * fs) / 100)
#
# print("...feature 3 - Sum")
# # 4 - Number of Peaks above STD
# signalPKS = WindowStat(signal, fs=fs, statTool='findPks', window_len=(win * fs) / 128)
# signalPKS2 = WindowStat(signal, fs=fs, statTool='findPks', window_len=(64 * fs) / 100)
#
# print("...feature 4 - Pks")
# # 5 - Amplitude Difference between successive PKS
# signalADF32 = WindowStat(signal, fs=fs, statTool='AmpDiff', window_len=(win * fs) / 128)
# # signalADF128 = WindowStat(signal, fs=fs, statTool='AmpDiff', window_len=(2*win*fs)/100)
#
# print("...feature 5 - AmpDif")
# # 6 - Medium Frequency feature
# signalMF = WindowStat(signal, fs=fs, statTool='MF', window_len=(32 * fs) / 128)
# print("...feature 6 - MF")
#
#
# def createSegmentedFile(sig, begin, end, sm=False):
#     sig_out = []
#     mean_segment_time = 0
#     for i in range(len(begin)):
#         if(sm==True):
#             s = smooth(sig[begin[i]:end[i]], window_len=250)
#             s = sig[begin[i]:end[i]] - s
#             sig_out += s.tolist()
#         else:
#             sig_out += sig[begin[i]:end[i]].tolist()
#         mean_segment_time += (end[i] - begin[i])
#
#     mean_segment_time = mean_segment_time/len(begin)
#
#     return sig_out, mean_segment_time
#
# def specgram(s, nfft, fs, ax):
#     Pxx, freqs, bins, im = ax.specgram(s, NFFT=nfft, Fs=fs, noverlap=120, cmap=plt.get_cmap('viridis'))
#
#     return Pxx, freqs, bins, im
#
# def mag(x, y, z):
#     return np.sqrt((x**2)+(y**2)+(z**2))
