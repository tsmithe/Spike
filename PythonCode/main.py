import InfoAnalysis
import spikePlot


#INFO ANALYSIS:
# ia = InfoAnalysis.InfoAnalysis(globals())
#ia = InfoAnalysis.InfoAnalysis()
#ia.singleCellInfoAnalysis(['Neurons_Epoch0_83853152', 'Neurons_Epoch0_83853152'],weightedAnalysis = 1,saveImage = True, showImage = True);
# ia.singleCellInfoAnalysis(['Neurons_Epoch0_'],weightedAnalysis = 1,saveImage = True, showImage = False);

#PLOT SPIKES
sp = spikePlot.SpikePlot();
sp.plotSpikes(saveImage = True, showImage = False);