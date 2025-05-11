##### README #####

Pipeline goes as follows: cv.py --> spark.py --> dnn.py 

demo.py is to show how different contrasts and brightness levels affect contour detection,

imgseg_demo.py shows the difference between an unprocessed frame, traditional computer vision methods, 
and a trained deep neural network on the binary image segmentation task. 

To run this yourself, do the following:

1. Run 'cv.py'

2. Step out of the frame for 3 seconds. If the background is bad, i.e. the original screen you see is not 
   completely covered by orange, press 'n' and step out of the frame for another 3 seconds.

3. Press 'r' to start recording. This saves the frame in your 'data/frames' and an annotated mask in 'data/masks'.

4. Once enough data is gathered, run 'spark.py' to start training the deep neural network model. The best model 
   is saved to the 'models' folder.

5. Run 'dnn.py' to see how well your model is trained! This should work like Zoom's background filter effect. 
   If enough data is trained on, i.e. different backgrounds and different poses, notice how the model's output is
   invariant to changes in background, unlike using traditional CV methods. This means that the model has learned 
   about you specifically!!!

NOTE- The included trained model is trained on data of just me. It may not work for you!
