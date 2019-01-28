import cv2
import numpy as np
import csv

def get_box_centre(box):
    return (int(box[0] + (box[2] / 2)), int(box[1] + (box[3] / 2)))

def draw_box(frame, box, colour):
    (x, y, w, h) = (int(v) for v in box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    
cv2.namedWindow("Display", cv2.WINDOW_NORMAL);
cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    
vs = cv2.VideoCapture("unwrapped_video_routes/route3_tripod.mp4")

multi_tracker = cv2.MultiTracker.create()

robot_tracker_index = None
trapezoid_tracker_indices = []

# Read initial frame
grab_success, original_frame = vs.read()
assert grab_success

trapezoid_tracker_points = np.zeros((4, 2), dtype=np.float32)

normalised_frame_points = np.asarray([[0.0, 1.0],
                                      [0.0, 0.0],
                                      [1.0, 0.0],
                                      [1.0, 1.0]], dtype=np.float32)

csv_track_file = open("route3.csv", "wb")
csv_track_writer = csv.writer(csv_track_file, delimiter=",")

trapezoid_tracker_config = cv2.FileStorage("median_flow_tracker_default.yml", cv2.FILE_STORAGE_READ)

# loop over frames from the video stream
while True:
    # Process events
    key = cv2.waitKey(1) & 0xFF
    
    # If all trackers are ready
    if (robot_tracker_index is not None and len(trapezoid_tracker_indices) == 4) or key == ord("a"):
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        grab_success, original_frame = vs.read()
        if not grab_success:
            break
        
    # Resize frame to fit window
    frame = cv2.resize(original_frame, (1920, 1080))
    
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    track_success, boxes = multi_tracker.update(frame)

    #if track_success:
    # If robot is being tracked, draw it in blue
    if robot_tracker_index is not None:
        draw_box(frame, boxes[robot_tracker_index], (0, 0, 255))
    
    # Draw trapezoid trackers in red
    for l in trapezoid_tracker_indices:
        draw_box(frame, boxes[l], (255, 0, 0))
        
    # Build numpy array of tracker points
    for i, l in enumerate(trapezoid_tracker_indices):
        centre = get_box_centre(boxes[l])
        trapezoid_tracker_points[i, :] = centre
        
    num_tracked_points = len(trapezoid_tracker_indices)
    if num_tracked_points > 1:
        poly_points = trapezoid_tracker_points[:num_tracked_points, :].astype(np.int32)
        cv2.polylines(frame, [poly_points], 
                        num_tracked_points == 4, (255, 0, 0))
        
    # If all tracking marks are made
    if num_tracked_points == 4 and robot_tracker_index is not None:
        perspective_transform = cv2.getPerspectiveTransform(trapezoid_tracker_points, normalised_frame_points)
        
        # Get position of bottom centre of robot
        robot_box = boxes[robot_tracker_index]
        robot_pos = np.asarray([[robot_box[0] + (robot_box[2] / 2), robot_box[1] + robot_box[3]]], dtype=np.float32)
        robot_pos = np.asarray([robot_pos])
        
        # Apply perspective transform
        robot_transformed_pos = cv2.perspectiveTransform(robot_pos, perspective_transform)[0][0]
        
        # Display
        time = vs.get(cv2.CAP_PROP_POS_MSEC)
        cv2.putText(frame, "%f: (%f, %f)" % (time, robot_transformed_pos[0], robot_transformed_pos[1]), 
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        csv_track_writer.writerow([time, robot_transformed_pos[0], robot_transformed_pos[1]])
    else:
        cv2.putText(frame, "Please place tracking markers with 'r' and 't' or press 'a' to advance to frame where robot is visible",
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        
    # Draw image
    cv2.imshow("Display", frame)
   
    
    
    if key == 27:
        break
    elif key == ord("r") and robot_tracker_index is None:
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Display", frame, fromCenter=False, showCrosshair=True)
        if any(box):
            # Add tracker to multi tracker
            robot_tracker_index = len(multi_tracker.getObjects())
            multi_tracker.add(cv2.TrackerCSRT_create(), frame, box)
    elif key == ord("t") and len(trapezoid_tracker_indices) < 4:
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Display", frame, fromCenter=True, showCrosshair=True)
        if any(box):
            # Add tracker to multi tracker
            trapezoid_tracker_indices.append(len(multi_tracker.getObjects()))
            
            tracker = cv2.TrackerMedianFlow_create()
            tracker.read(trapezoid_tracker_config.getFirstTopLevelNode())
            multi_tracker.add(tracker, frame, box)
    
    

#cv2.TrackerCSRT_create
