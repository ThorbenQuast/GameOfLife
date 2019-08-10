import numpy as np
import copy
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetPalette(ROOT.kStarryNight)
import sys
import os

class Universe:
	def __init__(self, epoch):
		self.field = None
		self.hist2d = None
		self.hist2d_path = None
		self.epoch = epoch

	def Shape(self):
		return self.field.shape

	def InitialiseFromFile(self, fpath):
		infile = open(fpath, "r")
		data = np.genfromtxt(infile)
		if data.shape[0]!=data.shape[1]:
			print "Shape of input field not a square!"
			print data.shape[0],"vs",data.shape[1]
			sys.exit()
		infile.close()
		self.field = data
		self.field = self.field.swapaxes(0, 1)

	def GetEntryCopy(self, nx, ny):
		return copy.deepcopy(self.field[nx][ny])

	def Copy(self, ref):
		self.field = np.zeros(ref.Shape())
		for nx in range(ref.Shape()[0]):
			for ny in range(ref.Shape()[1]):
				self.field[nx][ny] = ref.GetEntryCopy(nx, ny)

	def GetNActiveNeighbors(self, nx, ny):
		NActives = 0
		nmin_x = max(0, nx-1)
		nmax_x = min(nx+1, self.field.shape[0])
		nmin_y = max(0, ny-1)
		nmax_y = min(ny+1, self.field.shape[1])
		NActives = np.count_nonzero(self.field[nmin_x:nmax_x+1,nmin_y:nmax_y+1]==1.)
		if self.field[nx][ny]==1.:
			NActives-=1	
		return NActives

	def Evolve(self, ref):
		for nx in range(self.field.shape[0]):
			for ny in range(self.field.shape[1]):
				NActiveNeighbors = ref.GetNActiveNeighbors(nx, ny)
				active_pre = True if self.field[nx][ny]==1. else False
				active_post=False
				if active_pre and NActiveNeighbors==2:
					active_post = True
				elif active_pre and NActiveNeighbors==3:
					active_post = True
				elif not active_pre and NActiveNeighbors==3:
					active_post = True
				else:
					active_post = False
				self.field[nx][ny] = 1 if active_post else 0

	def Print(self, fpath):
		self.hist2d_path=fpath
		self.hist2d = ROOT.TH2F("h2_epoch%i"%self.epoch, "h2_epoch%i"%self.epoch, self.field.shape[0], -0.5, self.field.shape[0]-0.5, self.field.shape[1], -0.5, self.field.shape[1]-0.5)
		self.hist2d.SetTitle("Epoch: %i"%self.epoch)
		self.hist2d.GetXaxis().SetTitle("x (a.u.)")
		self.hist2d.GetYaxis().SetTitle("y (a.u.)")
		self.hist2d.GetXaxis().SetLabelSize(0)
		self.hist2d.GetYaxis().SetLabelSize(0)
		self.hist2d.SetStats(False)

		for nx in range(self.field.shape[0]):
			for ny in range(self.field.shape[1]):
				weight = 1. if self.field[nx][ny]==1. else 0.01
				self.hist2d.Fill(nx, self.field.shape[1]-1-ny, weight)

		canvas = ROOT.TCanvas("canvas_%i"%self.epoch, "canvas_%i"%self.epoch, 700, 640)
		self.hist2d.GetZaxis().SetRangeUser(0., 1.)
		self.hist2d.Draw("COL")
		canvas.Print(self.hist2d_path)

	def Cleanup(self):
		os.remove(self.hist2d_path)
		del self.hist2d
		del self.field