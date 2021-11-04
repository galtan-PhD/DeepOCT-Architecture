TRAIN_DIR = '../train'
TEST_DIR = '../test1'
DME_LBL = 0
NONDME_LBL = 1
DME = 'DME'
NONDME = 'non-DME'
LABEL_MAP = {
    DME: DME_LBL,      
    NONDME: NONDME_LBL 
}
DATA_SIZE = 18_000
IMG_SIZE = 224
SPLIT_RATIO = 0.8

BM3D_Sigma_SERI = 45
BM3D_Sigma_ZhangLab = 35
AdaptiveThreshold = 44