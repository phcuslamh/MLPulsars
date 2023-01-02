For users:
- Import the training data on line 16.
- Import the testing data on line 26.
- Change the number of hidden nodes, epoch, batch size, and learning rate in lines 37-41.
- The output would be given in colored graphs.
	Red: losses
	Green: accuracy (% of correct predictions)
	Blue: precision (100% * (true positive) / (true positive + false positive))
	Orange: recall (100% * (true positive) / (true positive + false negative))
	Purple: F1 scores (100% * (2 * precision * recall) / (precision + recall))
Note: ideally, losses should follow a decreasing trend, and the remaining outputs should follow an increasing trend.