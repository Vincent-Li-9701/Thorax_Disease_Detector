num_epochs = 6           # Number of full passes through the dataset
K = 3                 # Number of folds
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.001
weight_decay = 0.8

# train, validation, testing parameters
N = 50 # validate every N train batch
N_stop = 3 # stop if the model doesn't make new score in N validation trials

# data paths
image_dir = "./datasets/images/"
image_info = "./datasets/Data_Entry_2017.csv"
