from CNN_Detector import *
from xray_dataloader import *

def helper(model, test_loader):
  torch.cuda.empty_cache()
  gpu = torch.device("cuda")
  model = model.to(gpu)
  with torch.no_grad():

    TP, TN, FP, FN = np.zeros(14), np.zeros(14), np.zeros(14), np.zeros(14)

    for count, (images, labels) in enumerate(test_loader, 0):
      if count > 100:
        break
      images, labels = images.to(gpu), labels.type(torch.FloatTensor).to(gpu)
      outputs = model(images)
      labels = labels.cpu().numpy()
      prediction = torch.round(outputs).type(torch.FloatTensor).cpu().numpy()
      TP = TP + np.sum((labels == 1) * (prediction == 1), axis=0)
      TN = TN + np.sum((labels == 0) * (prediction == 0), axis=0)
      FP = FP + np.sum((labels == 0) * (prediction == 1), axis=0)
      FN = FN + np.sum((labels == 1) * (prediction == 0), axis=0)

  print("TP", TP)
  print("TN", TN)
  print("FP", FP)
  print("FN", FN)
  model = model.to(torch.device("cpu"))

def visualize(weight, filename):
  result = weight.numpy()
  result -= np.amin(result)
  result /= np.amax(result)
  result *= 255
  result = result.astype('int8')
  img = Image.fromarray(result, 'L')
  img.save(filename)


if __name__ == "__main__":
    batch_size = 16         
    seed = np.random.seed(1) 
    p_test = 1             

    transform_base = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
    extras = {"num_workers": 1, "pin_memory": True}
    
    image_dir = "./datasets/images_test/"
    image_info = "./datasets/Data_Entry_2017.csv"

    baseline = CNN_Detector()

    base_state = torch.load('baseline_state')
    baseline.load_state_dict(base_state)

    _, test_loader_base = create_3_split_loaders(batch_size, seed, transform=transform_base,
                                        p_test=p_test,
                                        shuffle=True, show_sample=False,
                                        extras=extras, image_dir=image_dir, image_info=image_info)

    print("baseline:")
    helper(baseline, test_loader_base)
