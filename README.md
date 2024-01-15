DETECTORS:
(the .mat files need to be in a ./files/ folder)

python detect_highGamma.py files/SUBJECT.mat
python detect_beta.py files/SUBJECT.mat

(creates a csv file with the oscillations in ./highgamma/ and ./beta/ folders)


ML:

(the high gamma files need to be in a ./files/ folder, the beta in ./files_beta/)

Exp 1:

python gaussian_method_exp1.py files/highGammaEncode.csv

Exp 2:

python gaussian_method_exp2.py files/highGammaEncode.csv files/highGamma_recall_random.csv

Exp3:

python3 gaussian_method_exp3.py files/highGamma_recall.csv threshold split

(threshold is used to divide the sessions in good/poor, split is the number of folds for the test-train split)




