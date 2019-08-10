import universe
import argparse
import tqdm
import random

output_dir = "./evolutions"

parser = argparse.ArgumentParser()
parser.add_argument("--NEpochs", help="number of epochs to be placed", type=int)
parser.add_argument("--InitialFile", help="file with the initial conditions", type=str)
parser.add_argument("--OutputFile", help="file with the initial conditions", type=str)
parser.add_argument("--NDims", help="dimension of random field", type=int, default=100)
args = parser.parse_args()

universes = []
universes.append(universe.Universe(0))

if args.InitialFile=="random":
	print "Generating random input..."
	tmppath = "initials/random.txt"
	with open(tmppath, "w") as infile:
		for nx in range(args.NDims):
			for ny in range(args.NDims):
				value = 1 if random.uniform(0., 100.) < 30. else 0
				infile.write("%i "%value)
			infile.write("\n")
	universes[0].InitialiseFromFile(tmppath)
	import os
else:
	universes[0].InitialiseFromFile(args.InitialFile)

print "Playing the game"
for nepoch in tqdm.tqdm(range(1, args.NEpochs+1)):
	universes.append(universe.Universe(nepoch))
	universes[nepoch].Copy(universes[nepoch-1])
	universes[nepoch].Evolve(universes[nepoch-1])

print "Making the video"
import imageio
with imageio.get_writer("%s/%s.mp4" % (output_dir, args.OutputFile), fps=10) as writer:
	for nepoch in range(args.NEpochs+1):
		impath = "%s/%s_%i.png" % (output_dir, args.OutputFile, nepoch)
		universes[nepoch].Print(impath)
		if nepoch==0:
			for i in range(10):
				writer.append_data(imageio.imread(impath))
		else:
			writer.append_data(imageio.imread(impath))
	writer.close()

print "Cleanup"

for nepoch in range(args.NEpochs+1):
	universes[nepoch].Cleanup()