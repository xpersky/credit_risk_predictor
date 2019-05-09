import math

import numpy as np

# normalization of the data in axes of 0 - 1

def normalize(arr):
    min_arr = min(arr)
    max_arr = max(arr) - min_arr
    return [(arr[i] - min_arr)/(max_arr) for i in range(len(arr))]

# normalization of the single value

def val_norm(val,min_val,max_val):
    if val >= max_val:
        return 1
    elif val <= min_val:
        return 0
    else:
        return (val - min_val)/max_val

# remaking values of array to our target values

def remake(arr,val1,target):
    return [target if arr[i] == val1 else arr[i] for i in range(len(arr))]

# sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivative

def sig_der(x):
    return x * (1 - x)

# euclidean distance

def eucDist(inst1 , inst2, dim):
    distance = 0
    for i in range(dim):
        distance += pow((inst1[i] - inst2[i]), 2)
    return math.sqrt(distance)

# pre clustering function to match each point to the nearest centroids and return clusters for points

def preCluster(dset,centroids):
    clusters = []
    for i in range(len(dset)):
        distances = []
        for x in range(len(centroids)):
            distances.append((eucDist(dset[centroids[x]],dset[i],3),x+1))
        _, cluster = min(distances)
        clusters.append(cluster)
    return clusters

# setting centroids based on means

def setCentroids(dset,clusters,centroids):
    new_centroids = []
    for i in range(len(centroids)):         # for each cluster
        label = i + 1
        distances = []
        for x in range(len(clusters)):      # we are computing distance between all of the points
            if clusters[x] == label:
                for y in range(x,len(clusters)):
                    if clusters[y] == label and not x == y:
                        distances.append((eucDist(dset[x],dset[y],3),x,y))

        distances.sort()    # then we sort it
    
        # and then we starting with means

        count = []

        # we need to get all info about points

        for item in distances:
            dist, first, second = item
            if not first in count:
                count.append((first,dist,1))
            else:
                index = count.index(first)
                _,distance, amount = count[index]
                distance += dist
                amount += 1
                count[index] = first,distance,amount
            if not second in count:
                count.append((second,dist,1))
            else:
                index = count.index(second)
                _, distance, amount = count[index]
                distance += dist
                amount += 1
                count[index] = second,distance,amount

        # and compute means by gained info

        means = []
        proportion = []     # also we will use a little bit of wages ( so we could see which point is near to most of others)

        for item in count:
            point, dist, amnt = item
            means.append((dist/amnt,amnt,point))

        means.sort()    # then we will just sort the means

        for item in means:
            _, amnt, _ = item
            proportion.append(amnt)

        amnt_sum = sum(proportion)
        wage = [ item/amnt_sum for item in proportion ] # and make some wages

        max_wage = max(wage)
        avg_wage = sum(wage)/len(wage)

        iterator = 0

        while True:
            if wage[iterator] - avg_wage > 0:   # so starting from minimum mean
                break                           # we are checking if wage is good enough
                                                # and then breaking the loop
            elif abs(wage[iterator] - avg_wage) < max_wage - avg_wage:
                break

            if iterator == len(wage) - 1:
                break
            
            iterator += 1

        if not iterator == len(wage) - 1:
            _, _, point = means[iterator]
        else:
            _, _, point = means[0]

        # then we just are taking points into new centroids

        new_centroids.append(point)

    return new_centroids

# classification function

def classify(dset,centroids):
    clusters = preCluster(dset,centroids)   # using preclaster function explained above
    while True:                             # setting new centroids
        new_centroids = setCentroids(dset,clusters,centroids)
        clusters = preCluster(dset,centroids)
        count = 0

        # checking if new centroids are the same if not, we are still going

        for i in range(len(centroids)):
            if centroids[i] == new_centroids[i]:
                count += 1 
        if len(centroids) == count:
            break

        # else we set centroids again and againg unitil we get it right

        centroids = new_centroids
        print("-"*96)
        print("Making another centroids".center(96))

    return clusters

# get the nearest cluster

def getCluster(inst,dset,centroids):
    distances = []
    for i in range(len(centroids)):
        distances.append((eucDist(dset[centroids[i]],inst,3),i+1))
    _, cluster = min(distances)
    return cluster
        
 


