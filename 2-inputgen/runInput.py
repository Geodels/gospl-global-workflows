import argparse
from plateinputs.model import Model as sim

# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="This is a simple entry to run plateInputs model.", add_help=True
)

parser.add_argument("-i", "--input", help="Input file name (YAML file)", required=True)
parser.add_argument(
    "-v",
    "--verbose",
    help="True/false option for verbose",
    required=False,
    action="store_true",
    default=False,
)


args = parser.parse_args()

# Reading input file
model = sim(args.input, args.verbose)

# Running through time
model.runProcesses()
