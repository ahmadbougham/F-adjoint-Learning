# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Lead train and test datasets
fashion_mnist_train = pd.read_csv("~/Documents/2023-2024/Mai_2024/Version_3Mai/fashion-mnist_train.csv")
fashion_mnist_test = pd.read_csv("~/Documents/2023-2024/Mai_2024/Version_3Mai/fashion-mnist_test.csv")
# Separate target and features for training set
y_train = fashion_mnist_train["label"]
x_train = fashion_mnist_train.drop(["label"], axis = 1)
# Separate target and features for training set
y_test = fashion_mnist_test["label"]
x_test = fashion_mnist_test.drop(["label"], axis = 1)
# Print shapes of target and features
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
#x_train = x_train.reshape((-1, 28*28))
x_train = np.array(x_train)
#x_test = x_test.reshape((-1, 28*28))
x_test = np.array(x_test)
y_train = np.array(y_train)
#x_test = x_test.reshape((-1, 28*28))
y_test = np.array(y_test)
# Sanity Check
x_train = x_train.reshape((-1, 28*28)).T
x_train = x_train.astype('float64')/255

x_test = x_test.reshape((-1, 28*28)).T
x_test = x_test.astype('float64')/255

y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray().T
y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray().T
#print(x_train.shape, y_train.shape)

# %%
############################### Defining Activation Function Classes and their methods! ##########################################
class Relu_Class:
  def activation(z):
    return np.maximum(0,z)
  def prime(z):
      z[z<=0] = 0
      z[z>0] = 1
      return z

class Leaky_Relu_Class:
  @staticmethod
  def activation(z):
    alpha = 0.1
    return np.where(z<=0,alpha*z,z)
  def prime(z):
    alpha = 0.1
    return np.where(z<=0,alpha,1)

class Sigmoid_Class:
  @staticmethod
  def activation(z):
      return 1 / (1 + np.exp(-z))
  def prime(z):
      return -Sigmoid_Class.activation(z)*Sigmoid_Class.activation(z)+ Sigmoid_Class.activation(z)

class tanh_Class:
  @staticmethod
  def activation(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
  def prime(z):
    return 1 - np.power(tanh_Class.activation(z), 2)

class softmax_Class:
  @staticmethod
  def activation(x):
    e = np.exp(x-np.max(x))
    s = np.sum(e, axis=1, keepdims=True)
    return e/s
  @staticmethod
  def prime(z):
      return -softmax_Class.activation(z)*softmax_Class.activation(z) + softmax_Class.activation(z)

############################### Defining the Softmax Loss (Cross_Entropy) Class and its related methods #######################################################
class Cross_Entropy:
  def __init__(self, activation_fn):
      self.activation_fn = activation_fn

  def activation(self, z):
    return self.activation_fn.activation(z)

  def loss(y_true, y_pred):
      epsilon=1e-12
      y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
      N = y_pred.shape[0]
      loss = -np.sum(y_true*np.log(y_pred+1e-9))/N
      return loss

  @staticmethod
  def prime(Y, AL):
      return AL - Y

  def delta(self, y_true, y_pred):
      return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)

# %%
class MultiLayerPerceptron:
  #Constructor
  def __init__(self, dimensions, activations, weight = "uniform"):
    """
    list of dimensions (input, hidden layer(s), output)
    list of activation functions (relu, softmax etc) (in the case of 1 hidden layer)
    """
    self.num_layers = len(dimensions)-1 #Tells us the number of layers in the MLP model
    
    self.loss = None      #init
    self.alpha = None   #init
    # Weights, biases and the activations will be initialized as dictionaries, with the key being the index and the value being the subsequent Matrix
    self.w = dict()
    #self.b = dict()
    self.activations = dict()
    self.lambd = None
    # initial values for the weights as well as the biases according to the number of layers we have in the model
    for i in range(1,self.num_layers+1):
      # Starting from index i=1, we initialize weights according to the weight parameter
      if weight == "zeros":
        self.w[i ] = np.zeros((dimensions[i],dimensions[i-1]+1))
      elif weight == "uniform":
        self.w[i] = np.random.uniform(-1,1, (dimensions[i],dimensions[i-1]+1))
      elif weight == "gaussian":
        self.w[i] = np.random.normal(0,1, (dimensions[i],dimensions[i-1]+1))
      elif weight == "xavier":
        std_dev = np.sqrt(2/(dimensions[i] + dimensions[i-1]))
        self.w[i] = np.random.normal(0,std_dev, (dimensions[i],dimensions[i-1]+1))
      elif weight == "kaiming":
        self.w[i] = np.random.normal(0,2/(dimensions[i]), (dimensions[i],dimensions[i-1]+1))
      elif weight == "orthogonal":
        H=np.random.randn(dimensions[i],dimensions[i-1]+1)
        U,S,V= np.linalg.svd(H, full_matrices=False)
        self.w[i ] =np.dot(U,V) 
      else:  
        self.w[i ] =np.random.randn(dimensions[i],dimensions[i-1]+1) / np.sqrt(dimensions[i])  #Weight matrix set to the |i| X |i+1|

      #Now we init the biases to 0's according to the appropriate dimension
      #self.b[i + 1] = np.zeros(dimensions[i + 1])

      #Now we init the activations (starting from 1, since 0 will have the x as input)
      self.activations[i] = activations[i-1] 

  def F_pass(self, x):
    """
    Allows us to give an input, and go through the whole network till we reach the outputs
    Returns a tuple containing the z's and the activations applied onto the z's according to each activation function we've provided
    """
    Yf = dict()  #Creating a dictionary for the z's
    Xf = dict()  #Creating a dictionary for the a's      ##Note, these dictionaries are for a single forward prop instance

    Xf[0] = np.vstack( (x,np.ones((1,x.shape[1])) ))   #Setting as the first activation, the x's since that is what we will use

    for i in range(1,self.num_layers+1):      #Starting from 1, as all the dictionaries are indexed by 1
      #Calculating the z's and the a's
      Yf[i] = np.dot(self.w[i], Xf[i-1])      #Applying the weights to the x's/prev activation to get the next set of values
      if i !=self.num_layers:
        Xf[i] = np.vstack((self.activations[i].activation(Yf[i]),np.ones((1,self.activations[i].activation(Yf[i]).shape[1])) ) )
      else:
        Xf[i] = self.activations[i].activation(Yf[i])   #We apply the next activation function on the z we just calculated and put it in the next activation layer to be
                                                          # calculated in the next iteration for z

    return (Yf, Xf)   #Returning the tuple of z's and a's (to be used in the back prop / for prediction)
  def Fstar_pass(self, x, y):
    """
    Compute the partial derivatives and the error for the final layer, does backpropogation, by propogating the gradients back through the network
    """
    Xstar = dict()  #Creating a dictionary for the z's
    Ystar = dict()  #Creating a dictionary for the a's      ##Note, these dictionaries are for a single forward prop instance
    (Ya, Xa) = self.F_pass(x)
    Xstar[self.num_layers] = self.loss.prime(y, Xa[self.num_layers])
    #Now we have the back propogation loop!!  ###################################################################### Very Important!!!!!
    for i in reversed(range(1, self.num_layers+1)):
      Ystar[i]=Xstar[i]* self.activations[i].prime(Ya[i])
      Xstar[i-1]= np.dot(self.w[i][:,:-1].T, Ystar[i]) 
    return (Ystar, Xstar)
    
  def Fstar_pass_dyn(self, x, y, tau=0.1):
    #
    (FY, FX) = self.F_pass(x)    
    (TY, TX) =  (FY, FX) 
    TY[self.num_layers] =(FX[self.num_layers]-y)*self.activations[self.num_layers].prime(FY[self.num_layers])  
    ##Now we have the back propogation loop!!  ###################################################################### Very Important!!!!!
    for l in reversed(range(1, self.num_layers+1)):        
        if l!=self.num_layers:                            
            for  k in range(100):                               
                dXl= TX[l]  - np.dot( self.w[l+1].T,TY[l+1])#Fstar[f'X*{l+1}']*d_sigma(TX[f'Y{l+1}'])
                TX[l]=TX[l]-tau*dXl
            TY[l]= TX[l][:-1,:]*(self.activations[l].prime(FY[l]))      
        else:     
             #    
            for  k in range(100):     
                dXl= TX[l] - TY[l]#Fstar[f'X*{l}']*d_sigma(TX[f'Y{l}'])
                TX[l]=TX[l]-tau*dXl 
            TY[l]= TX[l]*(self.activations[l].prime(FY[l])) 


        self.w[l] = self.w[l] - self.alpha *  np.dot(TY[l], TX[l- 1].T)
  
  
        reg_matrix_w = np.zeros_like(self.w[l])

        if self.regg == 1:
          self.w[l]=self.w[l]+ (self.lambd) * np.sign(self.w[l])
          pass
        elif self.regg == 2:
          self.w[l]= self.w[l]+ 2 * (self.lambd) * self.w[l] 
    ############################        Completed Back Propogation!!!

  def predict(self, x):
    """
    Feeds forward the x to get the output, and returns the argmax
    """
    (garbage, a) = self.F_pass(x)
    return np.argmax(a[self.num_layers], axis = 0)

  def evaluate_acc(self, yhat, y):  ##Used for evaluating the accuracy, given true y's and predicted y's
    return np.mean(yhat == y)

 #Setting up some variables which we skipped over in the initialization which are needed
   
  def fit(self, x_train, y_train, epochs, mini_batch_size, alpha, x_test, y_test, regularization = 0, lambd = 0):
      """
      main function used to fit the model using the training data, and then used to test using the evac function
      """
      self.alpha = alpha
      self.loss = Cross_Entropy(self.activations[self.num_layers])#Cross_Entropy
      # self.lambd = regularization() Need to fix this!!!!!!!             ######## FIX THIS
      if regularization == 0:
        #print("No Reg")
        self.regg = 0
        self.lambd = 0
      elif regularization == 1:      ##Doing L1 Regularization
        print("L1 Reg")
        self.regg = 1
        self.lambd = lambd

      else: #Do L2 Regularization       #Regularization = 2,3 4 etc anything
        print("L2 Reg")
        self.regg = 2
        self.lambd = lambd


      #Setting up some miscelaneous parameters for graphing!
      self.train_logger = []
      self.test_logger = []
      self.cost_logger = []

      ##Starting the SGD!!!

      for i in range(epochs):
        #Randomizing the data:
        m=x_train.shape[1]
        permutation = np.random.permutation(m)
        shuffled_X = x_train[:,permutation]
        shuffled_y = y_train[:,permutation]       
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(num_complete_minibatches+1):
        ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:,mini_batch_size*(k):mini_batch_size*(k+1)]
            mini_batch_y = shuffled_y[:,mini_batch_size*(k):mini_batch_size*(k+1)]
            #Given the indices, now we do a forward pass on them, and then a backward pass
           # (z_dict, a_dict) = self.F_pass(mini_batch_X)

            #Now using the values in z_dict and a_dict, we do a backward pass to propogate the gradients back to the layers
            self.Fstar_pass_dyn(mini_batch_X, mini_batch_y)
            #Done training now

        #Now we check the training and testing accuracy and run the evaluate_acc function
        training_acc = self.evaluate_acc(self.predict(x_train), np.argmax(y_train,axis = 0))
        testing_acc = self.evaluate_acc(self.predict(x_test), np.argmax(y_test,axis = 0))

        #training_acc = np.mean(self.predict(x_train) == np.argmax(y_train,axis = 1))
        #testing_acc = np.mean(self.predict(x_test) == np.argmax(y_test,axis = 1))

        #Logging the accuracies
        self.train_logger.append(training_acc)
        self.test_logger.append(testing_acc)

        # print results for monitoring while training
        print("Epoch {0} train data: {1} %".format(i, 100 * (training_acc)))
        print("Epoch {0} test data: {1} %".format(i, 100 * (testing_acc)))    

# %%

def plot_fig(data):
    fig,ax = plt.subplots(1,1,figsize=(9,7))
    plt.title("Local F-adjoint model accuracy",fontsize=20,fontweight="bold",pad=25)
    for series_name, y in data:
        x = [f'{xx}' for xx in np.arange((len(y)))]
        ax.plot(np.asarray(x, float), y, label=series_name)
    plt.xlabel("Iteration",fontsize=20,style="oblique",labelpad=10)
    plt.ylabel("Accuracy",fontsize=20,style="oblique",labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = plt.legend(prop={"size":12})
    legend.fontsize=18
    legend.style="oblique"
    frame  = legend.get_frame()
    frame.set_facecolor("1.0")
    frame.set_edgecolor("1.0")
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.tick_params(axis='both',which='minor',labelsize=18)
    fig.tight_layout()  
    fig.savefig("Fashion_local.png", format="png") # specify filetype explicitly   
    plt.show()
    plt.close()

if __name__ == "__main__":
    
    import math
    np.random.seed(4) 
  # create model
    MLP_xavier = MultiLayerPerceptron([784,  128, 10], [Sigmoid_Class ,Sigmoid_Class], weight = "xavier")
    MLP_xavier.fit(x_train, y_train, 1000, 128, 0.001, x_test, y_test, regularization = 0, lambd = 0)
    # Plot data
    data = [("Training", MLP_xavier.train_logger),("Testing", MLP_xavier.test_logger)]    
    plot_fig(data)
