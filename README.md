# Baby Kick Visualizer

I built this because I wanted to capture and visualize those amazing moments when a baby kicks inside the belly. If you've ever tried to record your baby's movements on video, you know how hard it can be to actually *see* the kicks when you play it back. The movements are subtle, and breathing makes the whole abdomen move constantly.

This app takes a video of a pregnant belly and uses computer vision to detect and highlight the kicks. It filters out the slow, rhythmic breathing motion and finds the quick, localized movements that are actually baby kicks. Then it overlays a heat map on the video so you can clearly see where and when the kicks happened.

## What it does

- Detects baby kicks in video recordings of a pregnant abdomen
- Filters out breathing movements (which look similar but are slower and more uniform)
- Creates a heat map overlay showing movement intensity
- Generates a timeline graph of all the motion throughout the video
- Lets you export the processed video with the visualization baked in

## How it works

The app uses something called "optical flow" - basically tracking how pixels move between frames. When the abdomen moves from breathing, the whole region moves together in a smooth, predictable pattern. But when the baby kicks, there's a sudden, localized bump that looks very different.

By analyzing the frequency, location, and pattern of movements, the algorithm can tell the difference between the two. Breathing is slow (around 12-20 breaths per minute) and moves the whole belly uniformly. Kicks are faster, happen in specific spots, and create sudden spikes in the motion data.

## Getting started

You'll need Python 3.9 or higher.

```bash
# Clone and enter the project
git clone https://github.com/yourusername/babybumpviz.git
cd babybumpviz

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## Using the app

1. Upload a video of the pregnant belly
2. Draw a box around the abdomen area (this helps the algorithm focus on what matters)
3. Adjust the sensitivity if needed - start with the defaults
4. Click "Start Analysis" and wait for it to process
5. Watch the result and click on the timeline to jump to specific kicks

## Tips for better results

- Good lighting makes a big difference. Natural light or a well-lit room works best.
- Keep the camera steady. A tripod or resting it on something helps a lot.
- Tight-fitting clothing on the belly (or bare skin) shows movements more clearly than loose fabric.
- Shorter videos process faster. 30-60 seconds is usually enough to capture several kicks.
- If you're not detecting kicks, try lowering the sensitivity slider.

## The settings explained

**Sensitivity** - How much motion counts as "significant". Higher values catch smaller movements but might pick up noise. Lower values only catch strong, obvious kicks.

**Magnitude Threshold** - The minimum size of movement to be considered a kick. If kicks are being missed, lower this. If you're getting false positives, raise it.

**Heat Map Intensity** - How visible the overlay is. Adjust based on your preference.

## What's under the hood

The core algorithm uses OpenCV's Farneback optical flow to track motion, scipy for signal filtering (to separate kick frequencies from breathing frequencies), and numpy for all the number crunching. The web interface is built with FastAPI on the backend and plain HTML/CSS/JavaScript on the frontend.

## Known limitations

- Very early pregnancies won't show visible kicks on the surface yet
- If there's a lot of camera movement, it can confuse the detection
- Loose clothing hides the movements
- Processing takes a while for longer videos

---

This was a personal project born from wanting to preserve those fleeting moments of pregnancy. Hope it's useful to other expecting parents too.
