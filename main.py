import numpy as np
from config import *
from CNN_Detector import *
from xray_dataloader import create_3_split_loaders 
import torchvision
from torchvision import transforms


def main(): 
    
  seed = np.random.seed(1) # Seed the random number generator for reproducibility
  transform = transforms.Compose([transforms.ToTensor()])
  # Check if your system supports CUDA
  use_cuda = torch.cuda.is_available()

  # Setup GPU optimization if CUDA is supported
  if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
  else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

  # Setup the training, validation, and testing dataloaders
  loaders, _ = create_3_split_loaders(
      batch_size, seed, transform=transform, p_test=0, shuffle=True, 
      show_sample=False, extras=extras, image_dir=image_dir, image_info=image_info)

  # Instantiate a ExperimentOneCNN to run on the GPU or CPU based on CUDA support
  model = CNN_Detector()
  model = model.to(computing_device)
  print("Model on CUDA?", next(model.parameters()).is_cuda)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  # Track the loss across training
  total_loss = []
  total_valid_loss = []
  avg_minibatch_loss = []
  avg_minibatch_valid_loss = []
     
  total_acc = []
  total_valid_acc = []
  avg_minibatch_acc = []
  avg_minibatch_valid_acc = []
        
  # best state of the network
  best_valid_loss = None
  best_state = model.state_dict()
  
  # Begin training procedure
  for poch in range(num_epochs // K):
      for epoch in range(K):

        batch_since_best = 0
        N_minibatch_loss = 0.0
        N_minibatch_acc = 0.0

        train_loaders = set(range(K)) - {epoch}
        val_loader = loaders[epoch]
        
        # Get the next minibatch of images, labels for training
        for index in train_loaders:
            for minibatch_count, (images, labels) in enumerate(loaders[index], 0):

              # Put the minibatch data in CUDA Tensors and run on the GPU if supported
              images, labels = images.to(computing_device), labels.to(computing_device)

              # Zero out the stored gradient (buffer) from the previous iteration
              optimizer.zero_grad()

              # Perform the forward pass through the network and compute the loss
              outputs = model(images)

              criterion = nn.BCEWithLogitsLoss(pos_weight= calc_loss_weights(labels)) # Combine the last logistic layer with fully connected layer
              loss = criterion(outputs, labels.type(torch.cuda.FloatTensor))

              train_predictions = torch.round(torch.sigmoid(outputs))
              #prediction = torch.round(outputs)
              result = train_predictions == labels
              
              TP, TN, FP, FN = 0, 0, 0, 0
              TP += torch.sum((labels == 1) * (train_predictions == 1)).item()
              TN += torch.sum((labels == 0) * (train_predictions == 0)).item()
              FP += torch.sum((labels == 0) * (train_predictions == 1)).item()
              FN += torch.sum((labels == 1) * (train_predictions == 0)).item()


              if((TP + FP) != 0 and (TP + FN) != 0):
                  print("Precision:", TP / (TP + FP), "Recall:", TP / (TP + FN))    

              # Automagically compute the gradients and backpropagate the loss through the network
              loss.backward()

              # Update the weights
              optimizer.step()

              # Add this iteration's loss to the total_loss
              total_loss.append(loss.item())
              N_minibatch_loss += loss

              # Add this iteration's acc to the total_acc
              result = result.type(torch.FloatTensor)
              accuracy = torch.sum(result) / (result.size()[0] * result.size()[1])
              total_acc.append(accuracy)
              N_minibatch_acc += accuracy

              if (minibatch_count) % N == 0:

                # Print the loss averaged over the last N mini-batches    
                N_minibatch_loss /= N
                print('Epoch %d, average minibatch %d loss: %.3f' %
                    (epoch + poch*K, minibatch_count, N_minibatch_loss))

                N_minibatch_acc /= N
                print('Epoch %d, average minibatch %d acc: %.3f' %
                    (epoch + poch*K, minibatch_count, N_minibatch_acc))

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

                avg_minibatch_acc.append(N_minibatch_acc)
                N_minibatch_acc = 0.0

                # cross validation
                with torch.no_grad():

                  N_minibatch_valid_loss = 0.0
                  N_minibatch_valid_acc = 0.0  

                  # Calculate Validation Loss and Accuracy
                  for valid_minibatch_count, (images, labels) in enumerate(val_loader, 0):

                    # Perform the forward pass through the network and compute the loss on validation set
                    images, labels = images.to(computing_device), labels.to(computing_device)
                    outputs = model(images)

                    valid_predictions = torch.round(torch.sigmoid(outputs))
                    result = valid_predictions == labels
                    result = result.type(torch.FloatTensor)
                    accuracy = torch.sum(result) / (result.size()[0] * result.size()[1])
                    total_valid_acc.append(accuracy.item())
                    N_minibatch_valid_acc += accuracy

                    loss = criterion(outputs, labels.type(torch.cuda.FloatTensor))
                    total_valid_loss.append(loss.item())
                    N_minibatch_valid_loss += loss

                # record the loss
                N_minibatch_valid_loss /= valid_minibatch_count
                N_minibatch_valid_acc /= valid_minibatch_count
                avg_minibatch_valid_loss.append(N_minibatch_valid_loss)
                avg_minibatch_valid_acc.append(N_minibatch_valid_acc)

                # early stopping
                if (best_valid_loss is None) or (best_valid_loss > N_minibatch_valid_loss):
                  batch_since_best = 0
                  best_valid_loss = N_minibatch_valid_loss
                  best_state = model.state_dict()
                else:
                  batch_since_best += 1

                if batch_since_best >= N_stop:
                  print("Early stopping!")
                  break            

                print('Epoch %d, average valid loss loss: %.3f' %
                  (epoch + poch * K, N_minibatch_valid_loss))
                print('Epoch %d, average valid acc: %.3f' %
                  (epoch + poch * K, N_minibatch_valid_acc))

                N_minibatch_valid_loss = 0.0
                N_minibatch_valid_acc = 0.0

        # reload the best state
        model.load_state_dict(best_state)
        print("Finished", epoch + poch * K, "epochs of training")

  print("Training complete after", epoch + poch * K, "epochs")


def calc_loss_weights(labels):
    """ A function used to calculate weights for BCE loss fucntion

    Params:
        -------
        - labeles: The matrix of the labels mini_batch_size * number_of_classes

    Returns:
        --------
        - weights: The weight vector for all classes
    
    """

    weights = torch.zeros([1, len(labels[1])]).type(torch.cuda.FloatTensor)
    labels_counts = torch.sum(labels, dim = 0) #the dimension should be 1 * label_dim
    total_counts = torch.sum(labels_counts) # the number of diseases occur in this batch

    for i in range(labels_counts.shape[0]): #this should be equal to 14 (numbers of classes
        if labels_counts[i] != 0:
            weights[0][i] = total_counts / labels_counts[i]
    
    return weights



if __name__ == "__main__":
    main()
