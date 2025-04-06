import pickle
from retrain import predict_glucose

if __name__=="__main__":
    # load
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    video_file = input("Enter path to fingertip video: ").strip('"')
    if video_file == "":
        print("Running Demo....")
        video_file = "samples/demo.mp4"
    predict_glucose(video_file, model)