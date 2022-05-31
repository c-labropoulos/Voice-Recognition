i will use Mel-frequency cepstral coefficients (MFCCs) asfeatures and Gaussian mixture models (GMMs) to classify them. First of all, download the
Voice Dataset:https://drive.google.com/file/d/1BPIH87RhrypLmsIdrSX5zoPOMezr17yq/view?usp=sharing It consists of a single word recorded by 9 different
people in the .wav format and it’s divided into train and test subset.

1. Import all necessary libraries
2. Training process - for each recording from the train subset:
a. Load recording 
b. Extract MFCC features - librosa.features.mfcc with n_mfcc=39 and sr set
accordingly to the input data;
c. Fit GMM to the extracted features - GaussianMixture with n_components=32
and random_state=0 
d. Add GMM to some collection (e.g. append to the list).
3. After successful training, classify the test recordings based on GMMs - for each test
recording:
a. Load and extract MFCC.
b. For each GMM in the collection, compute the score for MFCC extracted from
the test recording. GMM with the maximum score represents the detected
speaker.
