# Open this with PsychoPy
from psychopy import visual, core, event
import random
import serial

# --- Configuration ---
NUM_TRIALS = 50
WIN_SIZE =[1000, 600]  # not used anymore because of fullscreen
TARGET_OFFSET = 0.6  # Normalized units (0.6 is 60% of the way to the edge)
ACCURACY_RATE = 0.75

# --- Timing Configuration (Seconds) ---
IMAGERY_DURATION = 2.0   # How long the target flashes

# CHANGED: Replaced MOVE_DURATION with FEEDBACK_DURATION. 
# The cursor instantly jumps, then stays there for this duration 
# to give the brain a clean window to generate the ErrP.
FEEDBACK_DURATION = 1.0  
ITI_DURATION = 2.0       # Inter-trial interval

RUNS_PER_LOOP = 3 # each run is 50 trials (or whatever NUM_TRIALS is set to)
BREAK_DURATION = 60 # 60 second break between runs

# Trigger Hub Config - use COM3 on the Ahmanson desktop
PORT = 'COM3'
LEFT_TRIGGER = 1
RIGHT_TRIGGER = 2
REST_TRIGGER = 3

NO_ERROR_TRIGGER = 4
ERRP_TRIGGER_CAUSEDbyMOVEMENT = 5 # if the erroneous action was a movement, eg was supposed to move to the left/stay in rest and moved right
ERRP_TRIGGER_CAUSEDbyREST = 6 # if the erroneous action was rest, eg was supposed to move to the right/left and stayed in rest

# CHANGED: Added safe connection block
try:
    mmbts = serial.Serial()
    mmbts.port = PORT
    mmbts.open()
except Exception as e:
    print(f"Warning: Trigger hub not connected on {PORT}. Running without triggers.")
    mmbts = None

# --- Setup Window ---
win = visual.Window(color='black', units='norm', fullscr=True)

# --- Define Stimuli ---
cursor = visual.Circle(win, radius=0.05, fillColor='white', lineColor='white', pos=(0, 0))

target_left = visual.Circle(win, radius=0.10, fillColor=None, lineColor='white',
                            lineWidth=3, pos=(-TARGET_OFFSET, 0))
target_right = visual.Circle(win, radius=0.10, fillColor=None, lineColor='white',
                             lineWidth=3, pos=(TARGET_OFFSET, 0))

instr_text = visual.TextStim(win, text="", pos=(0, 0.6), height=0.12, wrapWidth=1.8)
break_text = visual.TextStim(win, text="", pos=(0, 0), height=0.12, wrapWidth=1.8)

# -- Session Loop --
for run in range(RUNS_PER_LOOP):
    # --- Run Loop ---
    for trial in range(NUM_TRIALS):

        if 'escape' in event.getKeys():
            break

        # 1. SETUP LOGIC
        targets_list =['left', 'right', 'rest']
        target_side = random.choice(targets_list)
        side_trigger = LEFT_TRIGGER if target_side == 'left' else RIGHT_TRIGGER

        if random.random() < ACCURACY_RATE:
            move_side = target_side
            outcome_trigger = NO_ERROR_TRIGGER
        else:
            erroneous_targets =[item for item in targets_list if item != target_side]
            move_side = random.choice(erroneous_targets)
            
            if move_side == 'rest':
                outcome_trigger = ERRP_TRIGGER_CAUSEDbyREST
            else:
                outcome_trigger = ERRP_TRIGGER_CAUSEDbyMOVEMENT
            

        # 2. PART A: IMAGERY INSTRUCTION (Text Only)
        instr_text.text = "Target about to flash!\nWhen it does, imagine moving the cursor to it."

        target_left.fillColor = None
        target_right.fillColor = None
        cursor.pos = (0, 0)

        target_left.draw()
        target_right.draw()
        cursor.draw()
        instr_text.draw()
        win.flip()

        core.wait(random.uniform(1.5, 2.5))

        # 3. PART B: IMAGERY ACTION (Flash) + LEFT/RIGHT trigger aligned to flash flip
        if target_side == 'left':
            target_left.fillColor = 'green'
            target_right.fillColor = None
        elif target_side == 'right':
            target_right.fillColor = 'green'
            target_left.fillColor = None
        else:
            target_right.fillColor = None
            target_left.fillColor = None

        target_left.draw()
        target_right.draw()
        cursor.draw()
        instr_text.draw()

        # Send cue trigger exactly on the flip that shows the flash
        if mmbts:
            win.callOnFlip(mmbts.write, bytes([side_trigger]))
        win.flip()

        core.wait(IMAGERY_DURATION)

        # 4. OUTCOME ACTION (Instant Jump) + outcome trigger aligned to screen flip
        if move_side == 'left':
            end_pos = -TARGET_OFFSET
        elif move_side == 'right':
            end_pos = TARGET_OFFSET
        else:
            end_pos = 0.0

        # ---- CHANGED: Instantaneous Movement ----
        # Instantly teleport the cursor to the final location
        cursor.pos = (end_pos, 0)

        target_left.draw()
        target_right.draw()
        cursor.draw()
        instr_text.draw()

        # Send outcome trigger exactly on the flip that initiates the instant jump
        if mmbts:
            win.callOnFlip(mmbts.write, bytes([outcome_trigger]))
        win.flip()

        # ---- Wait to let the ErrP form ----
        # Hold the visual feedback for the user to process the (correct/erroneous) outcome
        # This provides a clean EEG window for the ErrP!
        core.wait(FEEDBACK_DURATION)
        
        # Check if the user pressed escape during the feedback window
        if 'escape' in event.getKeys():
            win.close()
            core.quit()

        # Turn off flash
        target_left.fillColor = None
        target_right.fillColor = None

        target_left.draw()
        target_right.draw()
        cursor.draw()
        instr_text.draw()
        win.flip()

        core.wait(ITI_DURATION)
    
    win.flip()
    if run != RUNS_PER_LOOP - 1:
        timer = core.CountdownTimer(BREAK_DURATION)
        while timer.getTime() > 0:  # after 5s will become negative
            break_text.text = f"Break time!\nPlease come back in {int(timer.getTime())}s"
            break_text.draw()
            win.flip()
            
            if 'escape' in event.getKeys():
                win.close()
                core.quit()

# --- Cleanup ---
win.close()
core.quit()
if mmbts:
    mmbts.close()
