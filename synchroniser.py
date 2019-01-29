import cv2
import numpy as np
import csv

min_x = 620.0
min_y = 200.0

max_x = 880.0
max_y = 480.0

ms_sync = 0.0

# Open original video
vs = cv2.VideoCapture("raw_video_routes/early_afternoon/181_0001.mp4")

# Open raw route CSV
with open("unwrapped_video_routes/route3_jamie_raw.csv", "rb") as raw_route_csv_file, open("processed_routes/route3/route.csv", "wb") as output_csv_file:
    raw_route_csv = csv.reader(raw_route_csv_file, delimiter=",")
    output_csv = csv.writer(output_csv_file, delimiter=",")
    
    # Loop through rows
    for i, row in enumerate(raw_route_csv):
        assert len(row) == 3
        time = float(row[0])
        x = float(row[1])
        y = float(row[2])
        
        # Seek to correct time in video
        success = vs.set(cv2.CAP_PROP_POS_MSEC, time + ms_sync)
        assert success
        
        # Read frame
        grab_success, frame = vs.read()
        assert grab_success
        
        print(i)
        # Write frame to disk
        filename = "processed_routes/route3/%u.jpg" % i
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        transformed_x = min_x + (x * (max_x - min_x))
        transformed_y = max_y - (y * (max_y - min_y))
        
        output_csv.writerow([transformed_x, transformed_y, filename])
