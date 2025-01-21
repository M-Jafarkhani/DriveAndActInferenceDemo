import cv2
import numpy as np
import torch
import torchvision
import pickle as pkl
from torch.autograd import Variable
import os
import ProcessingUtilsMini
import I3D
from collections import OrderedDict

"""
Preforms a single inference forward pass given a video chunk (array of images)

Arguments:
    images: numpy array of images (video chunk), dimensionality TxHxWxC, first dimension is the time
    network: a pytorch model
    annotation_converter: an array of strings for mapping predicted IDs to class names (see load_model function, which reads the annotation converter)
    cuda_active (optional): default - True ( the model is moved to the GPU.)
    reset_transform (torchvision.transform, optional): default - None. Resets the default transform with the specified reset_transform, if it is not None.

Returns: top1_class, top1_class_conf, all_class_conf
    top1_class (str): predicted top1 class
    top1_class_conf (float): confidence of the predicted top1 class
    all_class_conf (dict): confidences for all classes as a dict (key is the class name, value is the confidence)

"""

def run_inference_on_video_chunk(images, network, annotation_converter, cuda_active=True, reset_transform=None):
    if (reset_transform):
        transform = reset_transform
    else:  # default transform 
        transform = torchvision.transforms.Compose([
            ProcessingUtilsMini.Rescale(size=(252, 256)), #Resize frames
            ProcessingUtilsMini.RandomSelect(n=32), #Randomly select n=32 frames if more frames provided 
            ProcessingUtilsMini.CenterCrop(height=224, width=224), #Crop frames to (224, 224)
            ProcessingUtilsMini.normalizeColorInputZeroCenterUnitRange(), #Scale pixel values to the range [-1, 1]
            ProcessingUtilsMini.ToTensor() #Convert the NumPy array to a PyTorch tensor
        ])

    images_transformed = transform(np.asarray(images)) #Converts the input images to a NumPy array and applies the preprocessing pipeline
    images_transformed = images_transformed.unsqueeze(0) #Shape after unsqueeze(0): (1, C, T, H, W)

    if cuda_active:
        images_transformed = Variable(images_transformed.cuda())
        outputs = network(images_transformed).cuda()  #pass the transformed input to the model 
        outputs = np.squeeze(outputs.data.cpu().numpy())
    else:
        images_transformed = Variable(images_transformed)
        outputs = network(images_transformed)
        outputs = np.squeeze(outputs.data.numpy()) #Converts predictions to a NumPy array and removes extra dimensions.

    out_class_id = np.argmax(outputs)   # Index of the class with the highest score
    top1_class_conf = np.max(outputs)   # Confidence of the highest score

    top1_class = annotation_converter[out_class_id] #Convert the out_class_id to a human-readable activity using the annotation_converter.
    all_class_conf = dict(zip(annotation_converter, outputs)) #dictionary mapping all class indices to their confidence scores

    return top1_class, top1_class_conf, all_class_conf 


"""
Loads n_frames from video file at filepath as a numpy array, starting at start_frame and returns it as a numpy array.
n_frames is 0 if all the frames should be taken
Note, the video segment should not be too large (be careful with the default parameters, when n_frames = 0, meanining all frames are loaded into memory).
"""


def load_video_segment(filepath, start_frame=0, n_frames=0, visualize=False, waitKey=100):
    cap = cv2.VideoCapture(filepath)

    if (start_frame > 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);

    # Interate over frames in video
    images = []

    count = 0
    if (n_frames > 0):
        video_length = n_frames
    else:
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame - 1

    while cap.isOpened():
        # Extract the frame
        ret, image = cap.read()
        images.append(image)

        if (visualize):
            cv2.imshow('Video Frame ', image)
            cv2.waitKey(waitKey)

        count = count + 1
        # If there are no more frames left
        if (count > video_length - 1):
            cap.release()
            break
        # print(count)

    return (images)


"""
Iterate through a video file frame by frame, compute activity predictions and visualize them in the video 
Maintains buffer frames, uses pretrained I3D model to predict activities for the buffered frames.
Anotates the frames.
Arguments:
    filepath: path for the video file
    network: a pytorch model
    annotation_converter: an array of strings for mapping predicted IDs to class names (see load_model function, whichreads the annotation converter)
    start_frame (default 0): first frame of the video segment, for which the predictions should be computed
    n_frames (default 0, depicting the complete video): number of frames, for which the predictions should be computed (0 if the complete video should be used)
    visualize : default - False. Whether the video with the prediciotn should be visualized using opencv
    waitKey : default - 100. Parameter for visualization. 100 means, visualization with 100 ms pause between the frames
    buffer_size : default - 32.  Size of the stored frame buffer. The prediction is done on the chunk of this size. This means, if the size is 32, the network prediction is computed from the last 32 frames.
    cuda_active : default - True
    frequency : default - 1. Number of frames, after which a new prediction is computed. 1 means, a prediction is done after every frame.
    video_path_out: default - None. If None, a the original video is re-written together with the prediction to the video_path_out
    out_fps: default - 15, output fps
    vidwriter: None -  optionally, one can already provide a vidwriter for the output video. If None, and video_path_out is not None, a new one is created.

"""


# n_frames is 0 if all the frames should be taken
def interate_video_and_predict(filepath, network, annotation_converter, start_frame=0, n_frames=0, visualize=False,
                               waitKey=100,
                               buffer_size=32, cuda_active=True, frequency=1, video_path_out=None, out_fps=15,
                               vidwriter=None):
    cap = cv2.VideoCapture(filepath)  #Opens the video file using OpenCV

    if (start_frame > 0):  #If start_frame is specified, skips frames until the start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);

    # Interate over frames in video
    images = []   #A buffer to store frames

    count = 0
    if (n_frames > 0):
        video_length = n_frames         #If n_frames is provided, processes exactly n_frames frames
    else:
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame - 1  #processes all frames from start_frame to the end of the video
    if (video_path_out):
        os.makedirs(os.path.dirname(video_path_out), exist_ok=True)
    while cap.isOpened(): # Extract the frame
        ret, image = cap.read()

        images.append(image)  #Adds the current frame to the images buffer
        if len(images) > buffer_size:
            images = images[len(images) - buffer_size:len(images)]  #Trims the buffer to ensure it contains at most buffer_size frames.

        if count % frequency == 0: # (frequency = 1, predictions are made for every frame)

            top1_class, top1_class_conf, all_class_conf = run_inference_on_video_chunk(images, network,
                                                                                       annotation_converter,                                                                           cuda_active=cuda_active)

            top1_class_conf_percent = round(100 * top1_class_conf, 2)

            fontColor = (60 + 100 - top1_class_conf_percent, 1.8 * top1_class_conf_percent, 0)

            image_without_text = image
            image = ProcessingUtilsMini.add_text_to_image(image, "Frame nr: {}".format(count),
                                                          bottomLeftCornerOfText=(10, 20), fontScale=0.85)
            image = ProcessingUtilsMini.add_text_to_image(image, "{}%".format(top1_class_conf_percent),
                                                          bottomLeftCornerOfText=(10, np.shape(image)[0] - 35),
                                                          fontColor=fontColor, lineType=1, fontScale=0.85,
                                                          font=cv2.FONT_HERSHEY_DUPLEX)
            image = ProcessingUtilsMini.add_text_to_image(image, "{}".format(top1_class),
                                                          bottomLeftCornerOfText=(10, np.shape(image)[0] - 15),
                                                          fontColor=fontColor,
                                                          lineType=1, fontScale=0.85, font=cv2.FONT_HERSHEY_DUPLEX)

            if (visualize):
                cv2.imshow('Video Frame ', image)
                cv2.waitKey(waitKey)

            if (video_path_out):

                if vidwriter is None:
                    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    vidwriter = cv2.VideoWriter(video_path_out, fourcc, out_fps,
                                                (np.shape(image)[1], np.shape(image)[0]))

                vidwriter.write(image)

        count = count + 1
        print("Current frame number: {}/{}".format(count, n_frames))
        # If there are no more frames left
        if (count > video_length - 1):

            cap.release()
            if vidwriter is not None:
                vidwriter.release()

            break


def load_model(trained_model_path, annotation_converter_path, cuda_active=True):
    annotation_converter = pkl.load(open(annotation_converter_path, 'rb'))
    # Load the network
    network = I3D.I3D(len(annotation_converter), modality='rgb')

    print("Loading trained model: %s" % (trained_model_path))
    if (cuda_active):
        network.load_state_dict(torch.load(trained_model_path))
    else:
        network.load_state_dict(torch.load(trained_model_path, map_location='cpu'))
    print("Loading trained model done. Number of classes: {}".format(len(annotation_converter)))

    if cuda_active:
        network = network.cuda()

    torch.set_grad_enabled(False)
    network.eval()

    return network, annotation_converter


def test_run_inference_on_video_chunk():
    cuda_active = True

    # Load the model
    trained_model_path = "./demo_models/MidLevel/AllActions/view_ids_1/supervised_models/I3D/n_input_frames_32balance_by_sampling_True_affine_True_pretrained_True/2019-03-27-13-10-39/best_model.pth"
    annotation_converter_path = "./demo_models/MidLevel/AllActions/view_ids_1/supervised_models/I3D/n_input_frames_32balance_by_sampling_True_affine_True_pretrained_True/2019-03-27-13-10-39/annotation_converter.pkl"
    network, annotation_converter = load_model(trained_model_path, annotation_converter_path, cuda_active=cuda_active)

    # Load the video chunk (numpy array)
    filepath_video = "/cvhci/data/activity/Pakos/final_dataset/pakos_videos/vp1/run1b_2018-05-29-14-02-47.ids_1.mp4"
    start_frame = 30000
    n_frames = 32
    video_chunk = load_video_segment(filepath_video, start_frame=start_frame, n_frames=n_frames)

    # Make a prediction
    print("Computing prediction for video {}, video chunk from frame {} to frame {}".format(filepath_video, start_frame,
                                                                                            start_frame + n_frames))
    top1_class, top1_class_conf, all_class_conf = run_inference_on_video_chunk(video_chunk, network,
                                                                               annotation_converter,
                                                                               cuda_active=cuda_active)

    # Print all predictions sorted by confidence
    ranks = np.argsort(list(all_class_conf.values()))[::-1]
    classes_sorted = np.asarray(list(all_class_conf.keys()))[ranks]
    conf_sorted = np.asarray(list(all_class_conf.values()))[ranks]

    for i in range(len(conf_sorted)):
        print("{}) {} - {}".format(i + 1, classes_sorted[i], conf_sorted[i]))


def test_interate_video_and_predict(filepath_video="./test_data/run1b_2018-05-29-14-02-47.kinect_color.mp4", 
                                    video_path_out = "./test_data/output.mp4",
                                    start_frame=100,
                                    n_frames=200,
                                    cuda_active = True):
    """
    Iteratively process and predict over a video using a pre-trained model.
    Improtant: if n_frames = 0 - the whole video is processed!

    Parameters:
    filepath_video (str): Path to the video file to be processed.
    start_frame (int): The frame number from where to start the prediction.
    n_frames (int): The number of frames to process and predict. Improtant: if 0 - the whole video is processed!
    Returns:
    None - Depending on the function's parameters, it may save a video with predictions or display the visualization.
    best_model.pth contains the pretrained weights for the I3D model
    ( weights were on the Drive&Act dataset and were fine-tuned after pretraining on Kinetics)
    """
    


    # Load the model
    trained_model_path = "./d emo_models/best_model.pth"  
    annotation_converter_path = "./demo_models/annotation_converter.pkl"
    network, annotation_converter = load_model(trained_model_path, annotation_converter_path, cuda_active=cuda_active)

    #Note: out_fps must be 15 for RGB videos!
    interate_video_and_predict(filepath_video, 
                               network=network, 
                               annotation_converter=annotation_converter,
                               start_frame=start_frame, 
                               n_frames=n_frames, visualize=False, waitKey=100,
                               buffer_size=32, cuda_active=cuda_active, 
                               frequency=1, video_path_out=video_path_out,
                               out_fps=15)


if __name__ == '__main__':

    test_interate_video_and_predict(filepath_video="./test_data/run1b_2018-05-29-14-02-47.kinect_color.mp4", 
                                    video_path_out = "./test_data/output_2.mp4", #Path to save the output video with predictions
                                    start_frame=0, 
                                    n_frames=0,  #Process the entire video
                                    cuda_active = False)

    print("Done.")
