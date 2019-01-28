import cv2
import numpy as np

def get_box_centre(box):
    return (int(box[0] + (box[2] / 2)), int(box[1] + (box[3] / 2)))

def draw_box(frame, box, colour):
    (x, y, w, h) = (int(v) for v in box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

def draw_line(frame, box_1, box_2, colour):
    # Get the centre of the two boxes
    box_centre_1 = get_box_centre(box_1)
    box_centre_2 = get_box_centre(box_2)
    
    cv2.line(frame, box_centre_1, box_centre_2, colour)

cv2.namedWindow("Display", cv2.WINDOW_NORMAL);
cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    
vs = cv2.VideoCapture("unwrapped_video_routes/route4_tripod.mp4")

multi_tracker = cv2.MultiTracker.create()

robot_tracker_index = None
trapezoid_tracker_indices = []

# Read initial frame
grab_success, frame = vs.read()
assert grab_success

# Resize frame to fit window
original_frame = cv2.resize(frame, (1920, 1080))

trapezoid_tracker_points = np.zeros((4, 2), dtype=np.float32)

normalised_frame_points = np.asarray([[0.0, 0.0],
                                      [0.0, 1.0],
                                      [1.0, 1.0],
                                      [1.0, 0.0]], dtype=np.float32)

# loop over frames from the video stream
while True:
    # If all trackers are ready
    if robot_tracker_index is not None and len(trapezoid_tracker_indices) == 4:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        grab_success, frame = vs.read()
        if not grab_success:
            break
    else:
        frame = original_frame
        
    # Resize frame to fit window
    frame = cv2.resize(frame, (1920, 1080))
    
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    track_success, boxes = multi_tracker.update(frame)

    if track_success:
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
            
        if num_tracked_points == 4:
            perspective_transform = cv2.getPerspectiveTransform(trapezoid_tracker_points, normalised_frame_points)
            
            if robot_tracker_index is not None:
                robot_pos = get_box_centre(boxes[robot_tracker_index])
                robot_pos = np.asarray([[robot_pos[0], robot_pos[1]]], dtype=np.float32)
                robot_pos = np.asarray([robot_pos])
                
           
                robot_transformed_pos = cv2.perspectiveTransform(robot_pos, perspective_transform)[0][0]
    
                cv2.putText(frame, "%f, %f" % (robot_transformed_pos[0], robot_transformed_pos[1]), 
                            (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        
        
    # Draw image
    cv2.imshow("Display", frame)
   
    # Process events
    key = cv2.waitKey(1) & 0xFF
    
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
            multi_tracker.add(cv2.TrackerCSRT_create(), frame, box)
    
    

#cv2.TrackerCSRT_create
