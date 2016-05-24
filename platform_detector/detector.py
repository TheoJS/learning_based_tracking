import cv2
import numpy as np

# Datapath to the training images
datapath = "C:/detecting_and_tracking/learning_based_tracking/platform_detector/PlatformData/TrainImages/"

# Number pos/neg pairs used for training
SAMPLES = 20

# Return the path to an image given a base path and a class name
def path(cls,i):
    return "%s%s%d).jpg"  % (datapath,cls,i+1)

# Utility functions:

# Obtain a FLANN matcher(fast approximate nearest neighbor search)
def get_flann_matcher():
  flann_params = dict(algorithm = 1, trees = 5)
  return cv2.FlannBasedMatcher(flann_params, {})

# Obtain a Bag Of Word extractor
def get_bow_extractor(extract, flann):
  return cv2.BOWImgDescriptorExtractor(extract, flann)

# Obtain a SIFT detector/extractor
def get_extract_detect():
  return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()

# Return features from an image
def extract_sift(fn, extractor, detector):
  im = cv2.imread(fn,0)
  return extractor.compute(im, detector.detect(im))[1]

# Extract BOW features    
def bow_features(img, extractor_bow, detector):
  return extractor_bow.compute(img, detector.detect(img))

#Main
def platform_detector():
  pos, neg = "pos (", "neg ("
  detect, extract = get_extract_detect()
  matcher = get_flann_matcher()
  print "building BOWKMeansTrainer..."
  bow_kmeans_trainer = cv2.BOWKMeansTrainer(1000)
  extract_bow = get_bow_extractor(extract, flann)

  # Trainer filling
  print "adding features to trainer"
  for i in range(SAMPLES):
    print i
    bow_kmeans_trainer.add(extract_sift(path(pos,i), extract, detect))
    bow_kmeans_trainer.add(extract_sift(path(neg,i), extract, detect))
    
  voc = bow_kmeans_trainer.cluster()
  extract_bow.setVocabulary( voc )



  # Train SVM only if not existing yet
  if os.path.isfile("%sTrainedSVM/SVMPlatform.xml" % (datapath)):
    # Build train data arrays from available training images
    traindata, trainlabels = [],[]
    print "adding to train data"
    for i in range(SAMPLES):
      print i
      traindata.extend(bow_features(cv2.imread(path(pos, i), 0), extract_bow, detect))
      trainlabels.append(1)
      traindata.extend(bow_features(cv2.imread(path(neg, i), 0), extract_bow, detect))
      trainlabels.append(-1)

    # Train the SVM
    print "not building SVM"
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_RBF)

    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

    # Save SVM for future use
    svm.save("%sTrainedSVM/SVMPlatform.xml" % (datapath))
  else:
    print "building SVM"
    svm.load("%sTrainedSVM/SVMPlatform.xml" % (datapath))
    

    
  return svm, extract_bow
