import cv2
import mediapipe as mp
import math
from collections import defaultdict
#here come initilization of the hands and drawing utils
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ------------ geometry helpers ------------
# These functions na will calculate distances, angles, and bounding boxes.
# Then give us geometric information  i.e, coordinates about the hand and its landmarks.
def bbox_and_size(lm):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = maxx - minx
    h = maxy - miny
    size = max(w, h) + 1e-6  # just to avoid zero small epsilon
    return (minx, miny, maxx, maxy), size

def dist2d(a, b):
    # just using Euclidian distance between 2 points
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)

def angle_to_up(ax, ay, bx, by):
    """ Calculate angle between vector a->b and the UP direction (0, -1).
    Useful for figuring out if the thumb is pointing upwards."""
    vx, vy = (bx - ax, by - ay)
    norm = math.hypot(vx, vy) + 1e-9
    ux, uy = (0.0, -1.0)  #up vector onez
    cosang = (vx*ux + vy*uy) / norm
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

# ------------ finger state logic ------------
#these functions checking the placement of fingures like check whether fingers are extended, curled, or in special poses.

def finger_extended_y(lm, tip_idx, pip_idx, margin=0.0):
    return lm[tip_idx].y < (lm[pip_idx].y - margin)

def finger_curled(lm, tip_idx, mcp_idx, pip_idx, hand_size):
    not_up = lm[tip_idx].y >= lm[pip_idx].y
    chord = dist2d(lm[tip_idx], lm[mcp_idx]) / hand_size
    return not_up and (chord < 0.35)

def thumb_extended(lm, hand_label, hand_size):
    lateral = (lm[4].x < lm[3].x) if hand_label == "Right" else (lm[4].x > lm[3].x)
    far_from_index_mcp = dist2d(lm[4], lm[5]) / hand_size > 0.30
    straight_len = dist2d(lm[4], lm[2]) / hand_size
    fairly_straight = straight_len > 0.35
    return lateral or far_from_index_mcp or fairly_straight

def thumb_up_pose(lm, hand_label, hand_size):
    ang = angle_to_up(lm[2].x, lm[2].y, lm[4].x, lm[4].y)
    horiz = abs(lm[4].x - lm[2].x) / hand_size
    return (ang < 35.0) and (horiz < 0.45)

def is_v_sign(states):
    return states['index'] and states['middle'] and (not states['ring']) and (not states['pinky'])

def is_open_palm(states):
    return all(states.values())

def is_fist(lm, hand_size):
    index_curled  = finger_curled(lm, 8, 5, 6, hand_size)
    middle_curled = finger_curled(lm, 12, 9, 10, hand_size)
    ring_curled   = finger_curled(lm, 16, 13, 14, hand_size)
    pinky_curled  = finger_curled(lm, 20, 17, 18, hand_size)
    long_curled = index_curled and middle_curled and ring_curled and pinky_curled

    near_index_mcp  = dist2d(lm[4], lm[5]) / hand_size < 0.22
    near_middle_mcp = dist2d(lm[4], lm[9]) / hand_size < 0.22
    thumb_folded = near_index_mcp or near_middle_mcp

    return long_curled and thumb_folded

def is_thumbs_up(lm, hand_label, hand_size):
    index_curled  = finger_curled(lm, 8, 5, 6, hand_size)
    middle_curled = finger_curled(lm, 12, 9, 10, hand_size)
    ring_curled   = finger_curled(lm, 16, 13, 14, hand_size)
    pinky_curled  = finger_curled(lm, 20, 17, 18, hand_size)
    long_curled = index_curled and middle_curled and ring_curled and pinky_curled

    thumb_ext = thumb_extended(lm, hand_label, hand_size)
    thumb_dir_up = thumb_up_pose(lm, hand_label, hand_size)

    return long_curled and thumb_ext and thumb_dir_up

def is_rock(lm, hand_label, hand_size):
    """Rock sign: index + pinky extended, middle + ring curled, thumb can be folded/extended."""
    index_ext = finger_extended_y(lm, 8, 6)
    pinky_ext = finger_extended_y(lm, 20, 18)
    middle_curled = finger_curled(lm, 12, 9, 10, hand_size)
    ring_curled   = finger_curled(lm, 16, 13, 14, hand_size)
    return index_ext and pinky_ext and middle_curled and ring_curled

def is_call_me(lm, hand_label, hand_size):
    """Call Me sign (🤙): thumb + pinky extended, others curled."""
    thumb_ext = thumb_extended(lm, hand_label, hand_size)
    pinky_ext = finger_extended_y(lm, 20, 18)
    index_curled  = finger_curled(lm, 8, 5, 6, hand_size)
    middle_curled = finger_curled(lm, 12, 9, 10, hand_size)
    ring_curled   = finger_curled(lm, 16, 13, 14, hand_size)
    return thumb_ext and pinky_ext and index_curled and middle_curled and ring_curled

# --- temporal smoothing ---
 #It is like We don’t want gestures to flicker on/off with every frame.
# hold_frames tracks how long a gesture has been held before confirming it.

hold_frames = defaultdict(int)
REQUIRED_FRAMES = 3

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=2) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark

                hand_label = "Right"
                if res.multi_handedness and i < len(res.multi_handedness):
                    hand_label = res.multi_handedness[i].classification[0].label

                _, hand_size = bbox_and_size(lm)

                margin = 0.02
                states = {
                    'thumb':  thumb_extended(lm, hand_label, hand_size),
                    'index':  finger_extended_y(lm, 8, 6, margin),
                    'middle': finger_extended_y(lm, 12, 10, margin),
                    'ring':   finger_extended_y(lm, 16, 14, margin),
                    'pinky':  finger_extended_y(lm, 20, 18, margin)
                }

                # Gesture checks
                v_now     = is_v_sign(states)
                palm_now  = is_open_palm(states)
                fist_now  = is_fist(lm, hand_size)
                thumb_now = is_thumbs_up(lm, hand_label, hand_size)
                rock_now  = is_rock(lm, hand_label, hand_size)
                callme_now = is_call_me(lm, hand_label, hand_size)

                # Keys
                key_v     = (i, 'V')
                key_palm  = (i, 'PALM')
                key_fist  = (i, 'FIST')
                key_thumb = (i, 'THUMBUP')
                key_rock  = (i, 'ROCK')
                key_callme = (i, 'CALLME')

                for k, now in [(key_v, v_now), (key_palm, palm_now), (key_fist, fist_now),
                               (key_thumb, thumb_now), (key_rock, rock_now), (key_callme, callme_now)]:
                    if now:
                        hold_frames[k] = min(hold_frames[k] + 1, REQUIRED_FRAMES)
                    else:
                        hold_frames[k] = max(hold_frames[k] - 1, 0)
               # this part to display detected gestures on the screen
                y0 = 40 + 160*i
                if hold_frames[key_thumb] >= REQUIRED_FRAMES:
                    cv2.putText(frame, f"{hand_label} Thumbs-Up 👍", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2); y0 += 30
                if hold_frames[key_v] >= REQUIRED_FRAMES:
                    cv2.putText(frame, f"{hand_label} V-Sign ✌️", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2); y0 += 30
                if hold_frames[key_palm] >= REQUIRED_FRAMES:
                    cv2.putText(frame, f"{hand_label} Open Hand 🖐️", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2); y0 += 30
                if hold_frames[key_fist] >= REQUIRED_FRAMES:
                    cv2.putText(frame, f"{hand_label} Fist ✊", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (147,20,255), 2); y0 += 30
                if hold_frames[key_rock] >= REQUIRED_FRAMES:
                    cv2.putText(frame, f"{hand_label} Rock 🤘", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,20,147), 2); y0 += 30
                if hold_frames[key_callme] >= REQUIRED_FRAMES:
                    cv2.putText(frame, f"{hand_label} Call Me 🤙", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,128,255), 2); y0 += 30

        cv2.imshow("MediaPipe Gestures – robust", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
