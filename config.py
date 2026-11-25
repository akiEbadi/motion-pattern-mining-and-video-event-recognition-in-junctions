"""All settings in one place. Change values here, not inside the code."""

# ---------------------------------------------------------------------------
# Step 1: video -> clips -> optical flow -> words (thesis chapter 4, step 1)
# ---------------------------------------------------------------------------
CLIP_LENGTH = 75            # frames per non-overlapping clip (74 optical-flow fields per clip)
GRID_ROWS = 10              # spatial cells per frame; the thesis experiments used 10 x 10
GRID_COLS = 10
DIRECTION_BINS = 8          # quantized motion directions (thesis Fig. 3-4)
MIN_FLOW_MAGNITUDE = 0.5    # px/frame; a cell whose mean flow is weaker emits no word (static)

# ---------------------------------------------------------------------------
# Step 2: activity discovery. HDP is replaced by a neural topic model.
# ---------------------------------------------------------------------------
TRAIN_CLIPS = 500           # thesis: the first 500 clips are used for learning, the rest for testing
MAX_ACTIVITIES = 32         # upper bound on the number of topics (the thesis HDP learned 32)
TOPIC_HIDDEN = 128
TOPIC_EPOCHS = 300
TOPIC_LR = 1e-3
TOPIC_BATCH = 32
TYPICAL_RATIO = 0.99        # thesis eq. (6): keep activities whose cumulative ratio stays below this
NOISE_THRESHOLD = 0.90      # thesis eq. (8): keep words up to this cumulative probability

# ---------------------------------------------------------------------------
# Step 3: event classes (traffic states). HMM-LDA is replaced by a hybrid neural HMM.
# ---------------------------------------------------------------------------
N_CLUSTERS = 4              # number of event classes (thesis: 4)
HMM_HIDDEN = 64
HMM_EM_ITERS = 15
HMM_MLP_STEPS = 100         # gradient steps for the emission network in each EM iteration
HMM_LR = 1e-3
SILENT_THRESHOLD = 0.08     # thesis eq. (14); not used for decisions (activity vectors are normalised per clip)

# ---------------------------------------------------------------------------
# Empty-junction class (replaces the silent-clip rule of thesis eq. 14)
# ---------------------------------------------------------------------------
# Empty junction: no vehicle moves through the junction during the clip. A cell is "active" when it
# moves in at least EMPTY_MIN_STEPS of the clip's 74 flow steps. The clip is empty when the junction
# area (rows/cols below, frame edges excluded) has at most EMPTY_MAX_INNER active cells and the whole
# frame at most EMPTY_MAX_FULL. Tuned on 46 clips checked by eye (45 correct); the area is specific
# to this camera and the 10 x 10 grid. A single car crossing is NOT empty: its direction decides.
EMPTY_MIN_STEPS = 4
EMPTY_INNER_ROWS = (2, 9)   # python slice: rows 2..8
EMPTY_INNER_COLS = (1, 9)   # cols 1..8
EMPTY_MAX_INNER = 1
EMPTY_MAX_FULL = 4

# Same seed everywhere so runs can be repeated.
RANDOM_SEED = 42
