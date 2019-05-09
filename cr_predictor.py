from func import normalize, val_norm, remake, sigmoid, sig_der, eucDist,preCluster, setCentroids, classify, getCluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("-"*96)   # that's just decorator for better visualisation 
print("Loading dataset".center(96))     
                   
""" so there we go loading csv file using pandas
    file must be in project directory (or just change path whatever)
    first column in our dataset is index column """

path = './creditrisk.csv'                                   
dataset = pd.read_csv(path,index_col=0)                     
print("-"*96)
print("Normalizing data".center(96))                 

"""  we need to normalize our data to be from 0 to 1 
    to prevent the neural network from favoring variables 
    that can be multiple of others
    function normalize is in the func.py file   """

age = normalize(dataset['Age'].values.tolist())             
cr_amnt = normalize(dataset['Credit amount'].values.tolist())
duration = normalize(dataset['Duration'].values.tolist())
risk = dataset['Risk'].values.tolist()

# For some reasons (eg. user input) we would need max and min values of the dataset 

age_min = min(dataset['Age'].values.tolist())
age_max = max(dataset['Age'].values.tolist())
cra_min = min(dataset['Credit amount'].values.tolist())
cra_max = max(dataset['Credit amount'].values.tolist())
dur_min = min(dataset['Duration'].values.tolist())
dur_max = max(dataset['Duration'].values.tolist())

""" our risk column is filled with good and bad values 
    so we need to remake that to 0 and 1 so sigmoid function 
    would give us an output """

risk = remake(risk,"good",1)
risk = remake(risk,"bad",0)

np.random.seed(1) 
print("-"*96)
print("Initializing centroids".center(96))

""" our centroids would be random at start, 
    then using some means we would select the best centroids 
    and then we could divide dataset into clusters """

centroids = np.random.randint(1,1001,size=10).tolist()

# At first we need to classify data so neurons would predict only similiar examples 

size = 1000
data = []

for i in range(size):
    current = [age[i],cr_amnt[i],duration[i]]
    data.append(current)    

clusters = classify(data,centroids) # you can find function classify in a file func.py

# let's make a visualisation of clustered data

print("-"*96)
print("Making visualisation of the input data".center(96))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# we will make 10 scatters on one plot each would have another color to see if data is well clustered

ages = [[] for i in range(10)]
cras = [[] for i in range(10)]
durs = [[] for i in range(10)]

for i in range(len(clusters)):
    ages[clusters[i]-1].append(age[i])
    cras[clusters[i]-1].append(cr_amnt[i])
    durs[clusters[i]-1].append(duration[i])

colors = ['red','green','blue','yellow','black','pink','orange','purple','tan','brown']

for i in range(len(colors)):
    ax.scatter(ages[i],cras[i],durs[i], c=colors[i], marker='o')

ax.set_xlabel('Age')
ax.set_ylabel('Credit Amount')
ax.set_zlabel('Duration')

plt.show()

# Let's begin with some machine learning

samples = 700   # that's our training size
precision = 0   # that's our training precision

weights = []    # we would need some weights too

# so at first we will make them random

for i in range(len(centroids)):
    curr_weights = 2 * np.random.random((3,1))-1    
    weights.append(curr_weights)

starting_iterations = 10_000   # 100k iterations is good enough => about 68.5+ % accuracy
underrating_proportion = 0.1    # losing 0.x iterations each 0.x size of clusters

""" The explaination of underrating is simple, 
    perceptron does not need that much iterations
    each time it has better weights to predict the value
    so that's some mechanism to prevent it from overtrain """

# to use the mechanism above we need to know 
# how many sets we have in clusters
# and how many sets we already trained

clusters_size = [0 for i in range(len(centroids))] 
clusters_done = clusters_size

# and of course the number of iterations that must be done to each cluster element

cluster_training_iterations = [starting_iterations for i in range(len(centroids))] 

for i in range(samples):
    clusters_size[clusters[i]-1] += 1
print("-"*96)
print("Training".center(96))

loading = ["█"*(i+1) for i in range(96)] # that's only for animation to see the training progress

print("Progress{}".format("-"*88))  # and that's loading decorator

# so here we go , we start training our perceptrons

for iteration in range(samples):

    current_in = []     # so firstly we need to make our input table
    current_out = []    # and our output table

    # then simply append the values from each set to that tables

    input_value = [age[iteration],cr_amnt[iteration],duration[iteration]] 
    current_in.append(input_value)  
    current_out.append(risk[iteration])

    # and then we will make numpy arrays, they are easier to use in this case

    train_in = np.array(current_in)
    train_out = np.array(current_out).T

    # There is a learning process 

    for itera in range(cluster_training_iterations[clusters[iteration]-1]):
        input_layer = train_in                                                          # our input layer is now numpy array we made above
        output_layer = sigmoid(np.dot(input_layer,weights[clusters[iteration]-1]))      # to predict output we use sigmoid function
        error = train_out - output_layer                                                # we need to compute the error
        adj = error * sig_der(output_layer)                                             # and then compute adjustment by using sigmoid derivative
        weights[clusters[iteration]-1] += np.dot(input_layer.T,adj)                     # and then just update our weights
    
    clusters_done[clusters[iteration]-1] += 1   # after each lerning process we add the set to the done of the clusters
    
    # there we subtract iterations of clusters to prevent from overtrain

    if clusters_done[clusters[iteration]-1] % int(clusters_size[clusters[iteration]-1] * underrating_proportion) == 0:
        cluster_training_iterations[clusters[iteration]-1] -= int(cluster_training_iterations[clusters[iteration]-1]*underrating_proportion)
    
    # to provide better user experience we have loading bar that shows our progress

    print("{}".format(loading[int((iteration/samples)*len(loading))]), end="\r")

    # Now we can round our predictions so we would see the real values (good = 1 and bad = 0)

    round_predict = 1 if float(output_layer[0]) > 0.5 else 0

    # And at the end we would compute the precision of our training

    precision += 1 if round_predict == train_out else 0

print("")
print("-"*96)
print("Binary precision of training is {:.2f}%".format(precision/samples*100).center(96))

# After the training we can start testing our neural network

real_prec = 0               # that's the precision of the test
test_size = 1000 - samples  
accuracy = 0                # and that would be mean of not rounded predicted values 
                            # so we would see the real precision of the predictions

for iteration in range(samples,1000):

    current_in = []         # as before we start from making input table
    current_out = []        # and output table

    # then append the values

    input_value = [age[iteration],cr_amnt[iteration],duration[iteration]] 
    current_in.append(input_value)
    current_out.append(risk[iteration])

    # then making numpy arrays

    test_in = np.array(current_in)
    test_out = np.array(current_out).T

    # and then making a prediction

    prediction = sigmoid(np.dot(test_in,weights[clusters[iteration]-1]))

    # then we can round prediction

    round_prediction = 1 if float(prediction) >= 0.5 else 0

    # and coumpute the test precision

    real_prec += 1 if round_prediction == test_out else 0

    # and now we can compute our accuracy

    accuracy += abs(test_out - prediction[0])

# and show the results of the training

print("-"*96)
print("Test has ended succesfully with binary precision of {:.2f}%".format(real_prec/test_size*100).center(96))
print("-"*96)

print("And with accuracy of {:.2f}%".format(100-(accuracy[0]/test_size*100)).center(96))
print("-"*96)

print("Weights are".center(96))
print("-"*96)

# and then display the weights 

for index,item in enumerate(weights):
    print("| Weight {} | {:>25} | {:>25} | {:>25} |".format(index,item.tolist()[0][0],item.tolist()[1][0],item.tolist()[2][0]))
    print("-"*96)

# and now we can make our predictions from input 
print("Enter your data to predict the Credit Risk".center(96))
print("-"*96)

cr_age = int(input("Age : "))
print("-"*96)

cr_amount = int(input("Credit Amount ($) : "))
print("-"*96)

cr_duration = int(input("Credit duration (months) : "))
print("-"*96)

# and now we need to normalize those

input_age = val_norm(cr_age,age_min,age_max)
input_amt = val_norm(cr_amount,cra_min,cra_max)
input_dur = val_norm(cr_duration,dur_min,dur_max)

# and do every step similiar to test one

input_value = [[input_age,input_amt,input_dur]]

to_predict = np.array(input_value)

cluster = getCluster(input_value[0],data,centroids)

result = sigmoid(np.dot(to_predict,weights[cluster-1]))

# now get the result of that

if result > 0.5:
    message = "GOOD"
elif result <= 0.5:
    message = "BAD"

# and print this out, its our prediction

print("-"*96)
print("As a result, your CREDIT RISK is".center(96))
print("-"*96)
print(message.center(96))
print("-"*96)

# thanks for lookin at it :)
