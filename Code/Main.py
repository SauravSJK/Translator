import os, argparse
import Make_TF_Data, Model, Predict, Utilities

def main(args):
	if args.strtopt == 'a' or input("Run Make_TF_Data.py? (y/n): ") == 'y':
		Make_TF_Data.main()
	if args.strtopt == 'a' or input("Run Model.py? (y/n): ") == 'y':
		Model.main()
	if args.strtopt == 'a' or input("Run Predict.py? (y/n): ") == 'y':
		Predict.main()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Option")
	parser.add_argument("-so", "--strtopt", type = str, default = "", choices = ["a", ""], help = "a -> Run All: Make, Build and Predict")
	args = parser.parse_args()
	main(args)