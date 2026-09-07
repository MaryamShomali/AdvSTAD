import argparse

parser = argparse.ArgumentParser(
    description='Time-Series Anomaly Detection',
    epilog='Example: python main.py --model AdvSTAD --dataset SMD --retrain. '
           'Set advstad.fusion in config.yaml to sum, concat, or cross_attention.',
)
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name, e.g. AdvSTAD, TranAD, or LSTM_Multivariate")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--run-name',
					type=str,
					default=None,
					help="optional experiment label used in tracked run names")
args = parser.parse_args()
