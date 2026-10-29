from collections import deque
import cv2
import numpy as np
import math

# ================= AIR CANVAS =================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ================= HSV RANGE FOR BLUE CAP =================
# Adjust these values if your blue cap is not detected.

LOWER_BLUE = np.array([90, 80, 40])
UPPER_BLUE = np.array([135, 255, 255])

# ================= DRAWING COLORS (BGR) =================

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (120, 120, 120)

current_color = GREEN
current_color_name = "GREEN"

BRUSH_SIZE = 6
MAX_JUMP_DISTANCE = 100
SMOOTHING = 0.45

drawing_layer = None
previous_center = None
smooth_center = None
last_button = None

# ================= BUTTON POSITIONS =================

CLEAR_BUTTON = (40, 60, 170, 125)
BLUE_BUTTON = (190, 60, 290, 125)
GREEN_BUTTON = (310, 60, 410, 125)
RED_BUTTON = (430, 60, 530, 125)
YELLOW_BUTTON = (550, 60, 670, 125)

buttons = [
    ("CLEAR", CLEAR_BUTTON, GRAY),
    ("BLUE", BLUE_BUTTON, BLUE),
    ("GREEN", GREEN_BUTTON, GREEN),
    ("RED", RED_BUTTON, RED),
    ("YELLOW", YELLOW_BUTTON, YELLOW)
]


def inside(point, rect):
    if point is None:
        return False

    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(image, rect, text, color, text_color=WHITE):
    x1, y1, x2, y2 = rect

    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), BLACK, 2)

    cv2.putText(
        image, text, (x1 + 10, y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        text_color, 2, cv2.LINE_AA
    )


def reset_stroke():
    global previous_center, smooth_center
    previous_center = None
    smooth_center = None


print("AIR CANVAS STARTED")
print("Tracking object: BLUE CAP")
print("Drawing colours: BLUE, GREEN, RED, YELLOW")
print("Move the blue cap over a colour button to select it.")
print("Move the cap in the drawing area to draw.")
print("Press Q to quit.")

kernel = np.ones((5, 5), np.uint8)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read webcam.")
            break

        frame = cv2.flip(frame, 1)

        if drawing_layer is None:
            # White permanent canvas
            drawing_layer = np.full_like(frame, 255)

        # ================= DETECT BLUE CAP =================

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        center = None
        radius = 0

        # Ignore small blue objects
        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) > 300
        ]

        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest)

            if radius > 8:
                moments = cv2.moments(largest)

                if moments["m00"] != 0:
                    center = (
                        int(moments["m10"] / moments["m00"]),
                        int(moments["m01"] / moments["m00"])
                    )

        # ================= SMOOTH TRACKING =================

        if center is not None:
            if smooth_center is None:
                smooth_center = (
                    float(center[0]), float(center[1])
                )
            else:
                smooth_center = (
                    smooth_center[0] + SMOOTHING *
                    (center[0] - smooth_center[0]),
                    smooth_center[1] + SMOOTHING *
                    (center[1] - smooth_center[1])
                )

            point = (
                int(smooth_center[0]),
                int(smooth_center[1])
            )

            # Blue circle shows the detected cap
            cv2.circle(frame, point, 14, BLUE, 2, cv2.LINE_AA)
            cv2.circle(frame, point, 4, WHITE, -1, cv2.LINE_AA)

            # ================= BUTTON SELECTION =================

            button_now = None

            for name, rect, color in buttons:
                if inside(point, rect):
                    button_now = name
                    break

            if button_now is not None:
                if button_now != last_button:
                    reset_stroke()

                    if button_now == "CLEAR":
                        drawing_layer[:] = WHITE
                        print("Canvas cleared.")

                    elif button_now in ("BLUE", "GREEN", "RED", "YELLOW"):
                        current_color_name = button_now
                        current_color = {
                            "BLUE": BLUE,
                            "GREEN": GREEN,
                            "RED": RED,
                            "YELLOW": YELLOW
                        }[button_now]

                        print("Selected drawing colour:", button_now)

                last_button = button_now
                reset_stroke()

            else:
                last_button = None

                # ================= DRAW SMOOTH LINES =================

                if previous_center is not None:
                    distance = math.dist(previous_center, point)

                    if distance <= MAX_JUMP_DISTANCE:
                        steps = max(1, int(distance / 2))
                        start = previous_center

                        for i in range(1, steps + 1):
                            t = i / steps

                            next_point = (
                                int(start[0] + t * (point[0] - start[0])),
                                int(start[1] + t * (point[1] - start[1]))
                            )

                            cv2.line(
                                drawing_layer,
                                previous_center,
                                next_point,
                                current_color,
                                BRUSH_SIZE,
                                cv2.LINE_AA
                            )

                            previous_center = next_point
                    else:
                        # Avoid drawing unwanted long lines
                        previous_center = point
                else:
                    previous_center = point

        else:
            # Cap not visible: stop current stroke, keep old drawing
            reset_stroke()
            last_button = None

        # ================= TRACKING WINDOW =================

        tracking = frame.copy()

        # Overlay the white canvas only where marks exist
        gray = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
        drawing_mask = cv2.threshold(
            gray, 250, 255, cv2.THRESH_BINARY_INV
        )[1]

        tracking[drawing_mask > 0] = drawing_layer[drawing_mask > 0]

        for name, rect, color in buttons:
            text_color = BLACK if name == "YELLOW" else WHITE
            draw_button(tracking, rect, name, color, text_color)

        cv2.putText(
            tracking,
            f"Current drawing colour: {current_color_name}",
            (25, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, current_color, 2, cv2.LINE_AA
        )

        # ================= PAINT WINDOW =================

        paint = drawing_layer.copy()

        cv2.imshow("Tracking", tracking)
        cv2.imshow("Paint", paint)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
