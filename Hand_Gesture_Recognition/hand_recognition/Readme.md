**Real-Time Hand Gesture Detection (MediaPipe + OpenCV)**

By , Siddartha Arekanti

A lightweight, real-time webcam app that detects common hand gestures using MediaPipe Hands and OpenCV. Hold a gesture for a moment and you’ll see an on-screen label once it’s detected.




**Why these technologies? (Technology Justification)**

- MediaPipe Hands
  - Tracks many landmarks per hand with high accuracy and works well on CPU in real-time.
  - Also there is handness (Left/Right) and robust tracking that also try to bare partial occurences and change in lighting.
  - Saves us from training a model from scratch like where we right all coordinates and geometric logic with position of fingers and hands now Mediapipe Hands while still allowing explainable geometric logic.

- OpenCV
  - Simple webcam capture, frame processing, drawing overlays, and UI (text/lines).
  - fast, and found that it will pairs nicely with MediaPipe.

> Alternatives like pure contour-based heuristics are brittle (lighting/skin tone/background). Training a full deep model is heavier and heavy weight here—MediaPipe gives a better spot of speed + accuracy + simplicity.

---
**What gestures are supported?**


1. Open Palm 🖐️  
2.  Fist ✊  
3. Thumbs-Up 👍  
4. V-Sign ✌️
5. Rock🤘
6. Call Me🤙 

Detection is temporally smoothed (requires a few consecutive frames) to reduce flicker.

---

**How does it work? (Gesture Logic Explanation)**

Everything is built on normalized, scale-aware geometry so it works for different hand sizes and distances:

- Hand size normalization  
  We compute a bounding box over the the landmarks and use `hand_size = max(width, height)`. All distances/thresholds are divided by `hand_size` to be scale-invariant.

- Finger state primitives
  - `finger_extended_y(tip, pip)`: tip above PIP joint ⇒ finger is extended (in image Y goes down, so “above” means smaller Y).
  - `finger_curled(tip, mcp, pip, hand_size)`: tip not above PIP and the normalized chord `dist(tip, mcp)/hand_size` is small ⇒ finger is folded.
  - `thumb_extended(...)`: checks lateral separation from the index side and straightness along the thumb chain.
  - `thumb_up_pose(...)`: compares the angle of the thumb vector against the UP direction; filters out horizontal thumbs.

- Open Palm 🖐️ 
  All five fingers extended (`index, middle, ring, pinky` via Y-test + thumb via lateral/straightness checks).

- Fist ✊
  Index/Middle/Ring/Pinky all curled, and thumb is folded over (close to index/middle MCPs). This prevents a curled hand with an extended thumb from being mislabeled.

- Thumbs-Up 👍  
  Index/Middle/Ring/Pinky all curled; thumb extended and the thumb direction angle is close to UP (within ~35°) and not overly horizontal. This fixes the common “thumbs-up looks like a fist” issue.

- V-Sign ✌️  
  Index and Middle extended, Ring and Pinky not extended.

-🤘 Rock Sign – Index + Pinky extended; Middle + Ring curled; Thumb curled or neutral.

-🤙 Call-Me Sign – Thumb extended out; Pinky extended down/out; Index + Middle + Ring curled.


> We also apply temporal smoothing with a small frame counter per gesture (`REQUIRED_FRAMES = 3`), so labels appear only when the gesture is stable for a few frames.


**Run Instructions**
1. Clone this repo or copy codes 
   - Git clone https://github.com/Siddartharekanti/Hand-Gesture-Recognition.git
2. Move into the folder where the project app.py is there
3. pip install -r requirements.txt
4. python app.py


