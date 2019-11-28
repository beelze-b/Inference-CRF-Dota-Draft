from pystruct.learners import OneSlackSSVM, NSlackSSVM, SubgradientSSVM, SubgradientSSVM
from pystruct.learners import StructuredPerceptron, LatentSSVM, SubgradientLatentSSVM
from pystruct.learners import PrimalDSStructuredSVM, FrankWolfeSSVM
from pystruct.datasets import load_snakes
from pystruct.utils import make_grid_edges, edge_list_to_features
from pystruct.models import EdgeFeatureGraphCRF
import pickle
import numpy as np
import argparse

def get_X_features(X_no_features_df, most_popular_heroes, popularityDict, hero_win_percentage,
                   filtered_matches, use_popularity = False, use_map = False, default_value = -1):
    X_features = []
    for index, x in enumerate(X_no_features_df):
        X_feature = []
        for hero in x:
            win_rate = []
            for hero_id in most_popular_heroes:
                if hero_id not in x:
                    try:
                        win_rate.append(hero_win_percentage[hero][hero_id])
                    except:
                        win_rate.append( default_value )
                else:
                    win_rate.append( default_value )
            features = win_rate

            if use_popularity:
                for hero in x:
                    popularity = features.append(popularityDict[hero])
            #if radiant won, then enemy team was dire (0)
            #othrwise radiant lost, and enemy team was radiant
            features.append( int(filtered_matches[index]['didRadiantWin'] == False) )
            X_feature.append(features)
        X_features.append(X_feature)

    return X_features

def get_edge_features(X_features):
    X_edge_features = []
    #we are buiilding the edge features in the following order
    edges = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [1, 3], [1, 4],
             [2, 3], [2, 4],
             [3, 4]]
    for X_feature in X_features:
        edge_features = []
        for index, edge in enumerate(edges):
            node1 = edge[0]
            node2 = edge[1]
            edge_feature = np.concatenate((X_feature[node1], X_feature[node2]))
            
            edge_features.append(edge_feature)
            X_edge_features.append(edge_features)
    return edges, X_edge_features

def main(usePopularity, useMap, defaultValue, learner):

    #list of most popular heroes
    most_popular_heroes = [
        14, # pudge
        26, # lion
        74, # invoker,
        84, # ogre_magi
        41, # faceless_void
        21, # windrunner / windranger
        7, # earthshaker
        104, # legion_commander
        9, # mirana
        44, # phantom_assassin
        22, # zeus
        93, # slark
        5, # crystal_maiden
        8, # juggernaut
        86, # rubick
        35, # sniper
        6, # drow_ranger
        101, # skywrath_mage
        25, # lina
        2, # axe
        114, # monkey king
        27, # shadow shaman
        23, # kunkka
        99, # bristleback
        30, # witch_doctor
        32, # riki
        34, # tinker
        75, # silencer
        1, # anti mage
        42, # skeletonking/wraithking
        50, # dazzle
        90, # keeper of the light
        106, # ember spirit
        62, # bounty hunter
        60, # night stalker
        54, # life stealer
        17, # storm spirit
        3, # bane
        37, # warlock
        47, # viper
        113, # arc warden
        64, # jakiro
        72, # gyro
        81 # chaos knight
    ]

    #load data
    with open('data/popularityDict.pkl', 'rb') as f:
        popularityDict = pickle.load(f)
        
    with open('data/hero_win_percentage.pkl', 'rb') as f:
        hero_win_percentage = pickle.load(f)
            
    with open('data/X_no_features_df.pkl', 'rb') as f:
        X_no_features_df = pickle.load(f)

    with open('data/Y_df.pkl', 'rb') as f:
        Y_df = pickle.load(f)

    with open('data/filtered_matches.pkl', 'rb') as f:
        filtered_matches = pickle.load(f)

    #featurize
    X_features = get_X_features(X_no_features_df, most_popular_heroes, popularityDict,
                                hero_win_percentage, filtered_matches,
                                use_popularity = usePopularity, 
                                use_map = useMap, 
                                default_value = defaultValue)
    edges, X_edge_features = get_edge_features(X_features)
    X_train_edge_features = np.array( [(np.array(X_features[i]), np.array(edges), np.array(X_edge_features[i]) ) \
                                   for i in range(len(X_no_features_df))] )
    print(X_train_edge_features[0][0].shape)
    print(X_train_edge_features[0][1].shape)
    print(X_train_edge_features[0][2].shape)
    Y = [[most_popular_heroes.index(i) for i in y] for y in Y_df]


    #train model
    inference = 'qpbo'
    crf = EdgeFeatureGraphCRF(inference_method=inference)
    #ssvm = learner(crf, inference_cache=50, C=0.01, tol=.01, max_iter=100)
    ssvm = learner(crf, C=0.01, tol=.01, max_iter=100)

    X_train = X_train_edge_features[:1800]
    Y_train = Y[:1800]
    X_test = X_train_edge_features[1800:]
    Y_test = Y[1800:]

    ssvm.fit(X_train, np.array( Y_train ))

    #evaluate
    Y_train_pred = ssvm.predict(X_train)
    
    accuracy = lambda Y_predicted, Y_actual: np.average([sum(Y_predicted[i] == Y_actual[i])/5 \
                                                         for i in range(len(Y_predicted))])
    print("Train accuracy: ", accuracy(Y_train_pred, Y_train))

    Y_test_pred = ssvm.predict(X_test)
    print("Test accuracy: ", accuracy(Y_test_pred, Y_test))

    trainFilename = "data/Y_train_predictions_popularity_" + str(usePopularity) \
                    + "_default_" + str(defaultValue) + ".pkl"
    testFilename = "data/Y_test_predictions_popularity_" + str(usePopularity) \
                   + "_default_" + str(defaultValue) + ".pkl"
    
    with open(trainFilename, 'wb') as f:
        pickle.dump(Y_train_pred, f)

    with open(testFilename, 'wb') as f:
        pickle.dump(Y_test_pred, f)

if __name__ == '__main__':
    usePopularity = True
    useMap = False
    defaultValue = -1
    learner = FrankWolfeSSVM

    print("usePopularity: ", usePopularity)
    print("defaultValue: ", defaultValue)
    print("learner: ", learner)

    main(usePopularity, useMap, defaultValue, learner)
