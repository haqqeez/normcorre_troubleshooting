These scripts work on motion-corrected video data from Normcorre (or any motion-corrected methods really). It is encouraged to run these on the uncorrected video data as well as a point of reference as to whether the motion-corrected videos are significantly changed from the original.

The code works best with a GUI when crop == True. Setting it to false insteead uses the whole frame, but it can be beneficial to crop and focus on specific regions (e.g., regions with lots of cells, few cells, blood vessels, noticeable bulging, etc.). 

Correlation with first frame is suggested as the best means to compare videos. But additional stability metrics exist in the test_stability.py script.

compare_methods.m generated frame by frame correltions and frame by frame differences that also be useful, but is computationally expensive.

For inquiries, please contact z.haqqee@gmail.com
