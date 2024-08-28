# L2

### sOPTICS

#./DbscanCEOs --n_points 70000 --n_features 784 --X "data/mnist_all_X" --alg sOptics --eps 1800 --minPts 50 --ker_n_features 1024 --n_proj 1024 --topK 5 --topM 50 --distance L2 --output y_dbscan --n_threads 4 --ker_sigma 2600 


### sDBSCAN

./sDbscan --n_points 70000 --n_features 784 --X "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/test/data/mnist_all_X" --eps 0.2 --minPts 50 --ker_n_features 1024 --n_proj 1024 --topK 5 --topM 50 --samplingProb 0.01 --distance Cosine --sclusterNoise  --output y_dbscan --n_threads 64 --verbose

#./DbscanCEOs --n_points 70000 --n_features 784 --X "data/mnist_all_X" --eps 1300 --minPts 50 --ker_n_features 1024 --n_proj 1024 --topK 5 --topM 50 --distance L2 --clusterNoise 0 --output y_dbscan --n_threads 64


