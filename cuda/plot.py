import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import imageio, os

NEPOCHS = -1
DIMENSIONX = -1
DIMENSIONY = -1
INFILE = "evolution.txt"

with open(INFILE) as infile:
    line = infile.readline()
    NEPOCHS = int(line.replace("\n", "").split(",")[0])
    DIMENSIONX = int(line.replace("\n", "").split(",")[1])
    DIMENSIONY = int(line.replace("\n", "").split(",")[2])

epochs, x, y = np.genfromtxt(open(INFILE), skip_header=1, delimiter=",", usecols=(0,1,2), unpack=True)
fpaths = []

for epoch in range(NEPOCHS):
    h2 = ROOT.TH2F("h2_epoch%i"%epoch, "h2_epoch%i"%epoch, DIMENSIONX, -0.5, DIMENSIONX-0.5, DIMENSIONY, -0.5, DIMENSIONY-0.5)
    x_selected = x[np.where(epochs==epoch)]
    y_selected = y[np.where(epochs==epoch)]
    for entry_index in range(len(x_selected)):
        h2.SetBinContent(int(x_selected[entry_index]+1), int(y_selected[entry_index]+1), 1.)
    
    h2.SetTitle("Epoch: %i"%epoch)
    h2.GetXaxis().SetTitle("x (a.u.)")
    h2.GetYaxis().SetTitle("y (a.u.)")
    h2.GetXaxis().SetLabelSize(0)
    h2.GetYaxis().SetLabelSize(0)
    h2.SetStats(False)

    canvas = ROOT.TCanvas("canvas_%i"%epoch, "canvas_%i"%epoch, 700, 640)
    h2.GetZaxis().SetRangeUser(-0.05, 1.)
    h2.Draw("COL")
    fpath = "epoch_%i.png" % epoch
    canvas.Print(fpath)
    fpaths.append(fpath)

# visualise training as video of test samples
with imageio.get_writer("evolution.mp4", mode='I') as writer:
    for fpath in fpaths:
        image = imageio.imread(fpath)
        writer.append_data(image)

# cleanup images
for filename in fpaths:
    os.remove(filename)
