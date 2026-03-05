from __future__ import annotations

import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from mne_lsl.stream import StreamLSL

from bci_worker import RawCSVRecorder
from config import (
    EEGConfig,
    LSLConfig,
    MentalCommandLabelConfig,
    MentalCommandModelConfig,
    MentalCommandTaskConfig,
)
from mental_command_worker import (
    EMAProbSmoother,
    StreamingIIRFilter,
    evaluate_cv_quality,
    filter_block,
    make_bandpower_classifier,
    make_fb_riemannian_classifier,
    split_windows,
)


def run_task(fname: str):
    lsl_cfg = LSLConfig()
    eeg_cfg = EEGConfig()
    label_cfg = MentalCommandLabelConfig()
    task_cfg = MentalCommandTaskConfig()
    model_cfg = MentalCommandModelConfig()

    code_neutral = int(label_cfg.neutral_code)
    code_c1 = int(label_cfg.command1_code)
    code_c2 = int(label_cfg.command2_code)
    code_to_name = {
        code_neutral: "Neutral",
        code_c1: label_cfg.command1_name,
        code_c2: label_cfg.command2_name,
    }

    stream_bufsize = max(30.0, task_cfg.register_duration_s + 10.0)
    stream = StreamLSL(
        bufsize=stream_bufsize,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    print(f"[LSL] Stream info: {stream.info}")

    available = list(stream.info["ch_names"])
    eeg_picks = [ch for ch in eeg_cfg.picks if ch in available and ch != lsl_cfg.event_channels]
    if len(eeg_picks) < 2:
        eeg_picks = [ch for ch in available if ch != lsl_cfg.event_channels]
    if len(eeg_picks) < 2:
        raise RuntimeError(f"Need >=2 EEG channels, found: {available}")

    stream.pick(eeg_picks)
    sfreq = float(stream.info["sfreq"])
    ch_names = list(stream.info["ch_names"])
    print(f"[LSL] Connected: sfreq={sfreq:.1f} Hz, channels={ch_names}")

    raw_csv_path = f"{fname}_mental_command_raw.csv"
    raw_recorder = RawCSVRecorder(filepath=raw_csv_path, ch_names=ch_names)
    raw_recorder.start()

    win = visual.Window(size=(1280, 760), color=(-0.08, -0.08, -0.08), units="norm", fullscr=False)
    title = visual.TextStim(win, text="Mental Command Trainer", pos=(0, 0.78), height=0.06, color=(0.9, 0.9, 0.9))
    cue = visual.TextStim(win, text="", pos=(0, 0.42), height=0.08, color=(0.9, 0.9, 0.9))
    status = visual.TextStim(win, text="", pos=(0, -0.7), height=0.045, color=(0.85, 0.85, 0.85))
    detected = visual.TextStim(win, text="", pos=(0, 0.27), height=0.06, color=(0.95, 0.95, 0.95))

    bar_w = 1.40
    bar_h = 0.16
    bar_y = -0.02
    bar_outline = visual.Rect(
        win, width=bar_w, height=bar_h, pos=(0, bar_y), lineColor=(0.8, 0.8, 0.8), fillColor=None, lineWidth=2,
    )
    center_line = visual.Line(win, start=(0, bar_y - bar_h / 2), end=(0, bar_y + bar_h / 2), lineColor=(0.8, 0.8, 0.8))
    left_fill = visual.Rect(win, width=0.001, height=bar_h - 0.01, pos=(0, bar_y), fillColor=(-0.3, 0.8, 0.95), lineColor=None)
    right_fill = visual.Rect(win, width=0.001, height=bar_h - 0.01, pos=(0, bar_y), fillColor=(0.95, 0.65, -0.2), lineColor=None)
    neutral_dot = visual.Circle(win, radius=0.015, pos=(0, bar_y), fillColor=(0.7, 0.7, 0.7), lineColor=None, edges=64)

    left_lbl = visual.TextStim(win, text=label_cfg.command1_name, pos=(-0.48, -0.2), height=0.05, color=(0.8, 0.9, 1.0))
    right_lbl = visual.TextStim(win, text=label_cfg.command2_name, pos=(0.48, -0.2), height=0.05, color=(1.0, 0.9, 0.75))
    neutral_lbl = visual.TextStim(win, text="Neutral", pos=(0, -0.28), height=0.045, color=(0.9, 0.9, 0.9))

    def update_bar(p_left: float, p_right: float, p_neutral: float):
        p_left = float(np.clip(p_left, 0.0, 1.0))
        p_right = float(np.clip(p_right, 0.0, 1.0))
        p_neutral = float(np.clip(p_neutral, 0.0, 1.0))

        max_half = bar_w / 2.0
        lw = max_half * p_left
        rw = max_half * p_right

        left_fill.width = max(lw, 0.001)
        left_fill.pos = (-lw / 2.0, bar_y)
        right_fill.width = max(rw, 0.001)
        right_fill.pos = (+rw / 2.0, bar_y)

        neutral_dot.radius = 0.015 + 0.04 * p_neutral
        gray = 0.35 + 0.6 * p_neutral
        neutral_dot.fillColor = (gray, gray, gray)

    def draw_frame():
        title.draw()
        cue.draw()
        detected.draw()
        bar_outline.draw()
        center_line.draw()
        left_fill.draw()
        right_fill.draw()
        neutral_dot.draw()
        left_lbl.draw()
        right_lbl.draw()
        neutral_lbl.draw()
        status.draw()
        win.flip()

    def poll_escape():
        if "escape" in event.getKeys():
            raise KeyboardInterrupt

    def tick_recorder():
        raw_recorder.update(stream)

    def wait_with_display(duration_s: float):
        clk = core.Clock()
        while clk.getTime() < duration_s:
            tick_recorder()
            poll_escape()
            draw_frame()

    def wait_for_space():
        while True:
            tick_recorder()
            draw_frame()
            keys = event.getKeys()
            if "space" in keys:
                return
            if "escape" in keys:
                raise KeyboardInterrupt

    def wait_for_block_decision() -> bool:
        """Return True if accepted, False if rejected/retry."""
        while True:
            tick_recorder()
            draw_frame()
            keys = event.getKeys()
            if "space" in keys:
                return True
            if "r" in keys:
                return False
            if "escape" in keys:
                raise KeyboardInterrupt

    def collect_block(duration_s: float) -> np.ndarray:
        chunks = []
        last_ts = None
        clk = core.Clock()
        while clk.getTime() < duration_s:
            tick_recorder()
            data, ts = stream.get_data(winsize=0.30, picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts = np.asarray(ts)
                if last_ts is None:
                    mask = np.ones_like(ts, dtype=bool)
                else:
                    mask = ts > float(last_ts)
                if np.any(mask):
                    chunks.append(np.asarray(data[:, mask], dtype=np.float32))
                    last_ts = float(ts[mask][-1])
            poll_escape()
            draw_frame()
        if len(chunks) == 0:
            return np.empty((len(ch_names), 0), dtype=np.float32)
        return np.concatenate(chunks, axis=1)

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    classifier = None
    class_index = None
    smoother = None
    model_windows: list[np.ndarray] = []
    model_labels: list[int] = []
    model_block_ids: list[int] = []
    reject_thresh = eeg_cfg.reject_peak_to_peak
    w_samples = int(round(task_cfg.train_window_s * sfreq))

    try:
        # ==============================================================
        # Calibration phase
        # ==============================================================
        cue.text = (
            "Calibration phase\n"
            f"Record Neutral + {label_cfg.command1_name} + {label_cfg.command2_name}\n\n"
            "Press SPACE to begin. ESC to quit."
        )
        status.text = "Use a consistent, repeatable mental strategy per command."
        detected.text = ""
        update_bar(0.0, 0.0, 1.0)
        wait_for_space()

        class_sequence = [
            (code_neutral, "Neutral"),
            (code_c1, label_cfg.command1_name),
            (code_c2, label_cfg.command2_name),
        ]
        n_triplets = int(task_cfg.n_register_blocks)
        cue.text = (
            "Calibration sequence\n\n"
            f"{n_triplets} cycles of: Neutral -> {label_cfg.command1_name} -> {label_cfg.command2_name}\n"
            "Press SPACE to start."
        )
        status.text = f"Each block is {task_cfg.register_duration_s:.0f}s, then you choose accept or redo."
        detected.text = ""
        update_bar(0.0, 0.0, 1.0)
        wait_for_space()

        for block_num in range(1, n_triplets + 1):
            trial_block_id = block_num - 1
            for seq_idx, (code, class_name) in enumerate(class_sequence):
                attempt = 0
                while True:
                    attempt += 1
                    cue.text = f"Cycle {block_num}/{n_triplets}: {class_name}"
                    status.text = "Press SPACE to start this block."
                    detected.text = f"Attempt {attempt}"
                    update_bar(0.0, 0.0, 1.0)
                    wait_for_space()

                    cue.text = f"Prepare: {class_name}"
                    status.text = "Get ready..."
                    detected.text = ""
                    wait_with_display(task_cfg.prep_duration_s)

                    cue.text = f"Perform: {class_name}"
                    status.text = f"Hold this mental state for {task_cfg.register_duration_s:.1f}s"
                    raw_block = collect_block(task_cfg.register_duration_s)

                    if raw_block.shape[1] < w_samples:
                        print(
                            f"[REG] {class_name} cycle {block_num} attempt {attempt}: "
                            f"not enough data ({raw_block.shape[1]} samples), retrying"
                        )
                        cue.text = f"{class_name} block too short"
                        status.text = "Not enough EEG samples. Press SPACE to retry."
                        detected.text = ""
                        update_bar(0.0, 0.0, 1.0)
                        wait_for_space()
                        continue

                    # Filter the entire block continuously, THEN split into windows.
                    filtered_block = filter_block(raw_block, eeg_cfg, sfreq)
                    windows = split_windows(
                        block=filtered_block,
                        sfreq=sfreq,
                        window_s=task_cfg.train_window_s,
                        step_s=task_cfg.train_window_step_s,
                    )
                    if windows.shape[0] == 0:
                        cue.text = "No windows extracted"
                        status.text = "Press SPACE to retry this block."
                        detected.text = ""
                        update_bar(0.0, 0.0, 1.0)
                        wait_for_space()
                        continue

                    # Artifact rejection per window
                    n_total = int(windows.shape[0])
                    block_windows: list[np.ndarray] = []
                    for w in windows:
                        if reject_thresh is not None and float(np.ptp(w, axis=-1).max()) > reject_thresh:
                            continue
                        block_windows.append(w)
                    n_rejected = n_total - len(block_windows)

                    if len(block_windows) == 0:
                        cue.text = "No usable windows in this block"
                        status.text = f"All {n_total} windows exceeded artifact threshold. Press SPACE to retry."
                        detected.text = ""
                        update_bar(0.0, 0.0, 1.0)
                        wait_for_space()
                        continue

                    cue.text = f"Review: cycle {block_num}/{n_triplets} {class_name}"
                    status.text = (
                        f"{len(block_windows)} usable, {n_rejected} artifact-rejected, {n_total} total\n"
                        "SPACE = accept block, R = reject and redo"
                    )
                    detected.text = "Did you maintain the intended mental state?"
                    update_bar(0.0, 0.0, 1.0)
                    is_accepted = wait_for_block_decision()
                    if not is_accepted:
                        print(
                            f"[REG] {class_name} cycle {block_num} attempt {attempt}: user rejected"
                        )
                        cue.text = f"Redo: cycle {block_num}/{n_triplets} {class_name}"
                        status.text = "Press SPACE when ready to record again."
                        detected.text = ""
                        update_bar(0.0, 0.0, 1.0)
                        wait_for_space()
                        continue

                    for w in block_windows:
                        model_windows.append(w)
                        model_labels.append(int(code))
                        model_block_ids.append(trial_block_id)
                    print(
                        f"[REG] Accepted {class_name} cycle {block_num}/{n_triplets}: "
                        f"{len(block_windows)} usable, {n_rejected} rejected (attempt {attempt})"
                    )
                    break

                if not (block_num == n_triplets and seq_idx == len(class_sequence) - 1):
                    cue.text = f"Accepted: cycle {block_num}/{n_triplets} {class_name}"
                    status.text = "Press SPACE when ready for the next block."
                    detected.text = ""
                    update_bar(0.0, 0.0, 1.0)
                    wait_for_space()

        # ==============================================================
        # Model training — evaluate both pipelines, pick the better one
        # ==============================================================
        if len(model_labels) == 0:
            raise RuntimeError("No registration windows collected")

        counts = Counter(model_labels)
        for c in (code_neutral, code_c1, code_c2):
            if counts[c] < model_cfg.min_per_class_for_cv:
                raise RuntimeError(
                    f"Class {code_to_name[c]} has {counts[c]} samples; need at least {model_cfg.min_per_class_for_cv}"
                )

        X_train = np.stack(model_windows, axis=0)
        y_train = np.array(model_labels, dtype=int)
        block_ids = np.array(model_block_ids, dtype=int)

        cue.text = "Fitting classifiers..."
        status.text = "Evaluating Filter-Bank Riemannian and Log Band Power pipelines."
        detected.text = ""
        update_bar(0.0, 0.0, 1.0)
        draw_frame()

        fb_pipe = make_fb_riemannian_classifier(model_cfg, sfreq)
        bp_pipe = make_bandpower_classifier(model_cfg, sfreq)

        fb_quality = evaluate_cv_quality(
            X_train, y_train, block_ids, fb_pipe,
        )
        bp_quality = evaluate_cv_quality(
            X_train, y_train, block_ids, bp_pipe,
        )

        if fb_quality.balanced_accuracy >= bp_quality.balanced_accuracy:
            chosen_name = "Filter-Bank Riemannian"
            quality = fb_quality
            classifier = fb_pipe
        else:
            chosen_name = "Log Band Power"
            quality = bp_quality
            classifier = bp_pipe

        classifier.fit(X_train, y_train)
        class_index = {int(c): i for i, c in enumerate(classifier.named_steps["clf"].classes_)}
        smoother = EMAProbSmoother(
            alpha=task_cfg.live_display_smoothing_alpha,
            n_classes=len(classifier.named_steps["clf"].classes_),
        )

        np.save(f"{fname}_mental_command_windows.npy", X_train)
        np.save(f"{fname}_mental_command_labels.npy", y_train)
        with open(f"{fname}_mental_command_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        cue.text = (
            "Calibration complete\n\n"
            f"FB Riemannian CV: {fb_quality.balanced_accuracy:.1%} bal acc\n"
            f"Log Band Power CV: {bp_quality.balanced_accuracy:.1%} bal acc\n\n"
            f"Selected: {chosen_name}\n"
            f"  Balanced acc: {quality.balanced_accuracy:.2%}  |  Macro F1: {quality.macro_f1:.2%}\n"
            f"  Neutral: {quality.per_class_accuracy[str(code_neutral)]:.0%}"
            f"  {label_cfg.command1_name}: {quality.per_class_accuracy[str(code_c1)]:.0%}"
            f"  {label_cfg.command2_name}: {quality.per_class_accuracy[str(code_c2)]:.0%}\n"
            f"Samples: {quality.n_samples}"
        )
        status.text = (
            f"Neutral={counts[code_neutral]}  "
            f"{label_cfg.command1_name}={counts[code_c1]}  "
            f"{label_cfg.command2_name}={counts[code_c2]}\n"
            "Press SPACE for live mode."
        )
        detected.text = ""
        update_bar(0.0, 0.0, 1.0)
        wait_for_space()

        # ==============================================================
        # Live feedback mode
        # ==============================================================
        cue.text = "Live mental command practice"
        status.text = (
            "Think Neutral or either command and watch the bar.\n"
            "Keys: 1=set target to command1, 2=command2, 0=neutral, ESC=exit."
        )
        detected.text = ""
        update_bar(0.0, 0.0, 1.0)

        target_text = "Target: none"
        target_stim = visual.TextStim(win, text=target_text, pos=(0, 0.62), height=0.05, color=(0.9, 0.9, 0.9))
        pred_clock = core.Clock()
        session_clock = core.Clock()
        p_vec = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

        live_filter = StreamingIIRFilter.from_eeg_config(
            eeg_cfg=eeg_cfg, sfreq=sfreq, n_channels=len(ch_names)
        )
        live_buffer = np.empty((len(ch_names), 0), dtype=np.float32)
        live_buffer_keep_n = int(round((task_cfg.train_window_s + task_cfg.live_filter_context_s) * sfreq))
        stream_pull_s = max(0.20, task_cfg.live_update_interval_s * 2.0)
        last_live_ts: float | None = None

        while session_clock.getTime() < task_cfg.live_duration_s:
            tick_recorder()
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "1" in keys:
                target_text = f"Target: {label_cfg.command1_name}"
            elif "2" in keys:
                target_text = f"Target: {label_cfg.command2_name}"
            elif "0" in keys:
                target_text = "Target: Neutral"

            data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts = np.asarray(ts)
                if last_live_ts is None:
                    mask = np.ones_like(ts, dtype=bool)
                else:
                    mask = ts > float(last_live_ts)
                if np.any(mask):
                    x_new = np.asarray(data[:, mask], dtype=np.float32)
                    last_live_ts = float(ts[mask][-1])
                    x_new_filt = live_filter.process(x_new)
                    live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
                    if live_buffer.shape[1] > live_buffer_keep_n:
                        live_buffer = live_buffer[:, -live_buffer_keep_n:]

            if pred_clock.getTime() >= task_cfg.live_update_interval_s:
                pred_clock.reset()
                if live_buffer.shape[1] >= w_samples:
                    x_win = live_buffer[:, -w_samples:]
                    p_raw = classifier.predict_proba(x_win[np.newaxis, ...])[0]
                    p_vec = smoother.update(p_raw)

            left_p = p_vec[class_index[code_c1]]
            right_p = p_vec[class_index[code_c2]]
            neutral_p = p_vec[class_index[code_neutral]]
            update_bar(left_p, right_p, neutral_p)

            best_idx = int(np.argmax(p_vec))
            best_code = int(classifier.named_steps["clf"].classes_[best_idx])
            best_conf = float(p_vec[best_idx])
            if best_code == code_neutral or best_conf < task_cfg.min_confidence_to_show:
                detected.text = f"Detected: Neutral ({neutral_p:.2f})"
            else:
                detected.text = f"Detected: {code_to_name[best_code]} ({best_conf:.2f})"

            title.draw()
            target_stim.text = target_text
            target_stim.draw()
            cue.draw()
            detected.draw()
            bar_outline.draw()
            center_line.draw()
            left_fill.draw()
            right_fill.draw()
            neutral_dot.draw()
            left_lbl.draw()
            right_lbl.draw()
            neutral_lbl.draw()
            status.draw()
            win.flip()

        cue.text = "Live session complete"
        status.text = "Press ESC to close."
        detected.text = ""
        while True:
            tick_recorder()
            draw_frame()
            if "escape" in event.getKeys():
                break

    except KeyboardInterrupt:
        print("\nSession interrupted.")
    finally:
        raw_recorder.stop()
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass


def _sanitize_participant_name(raw_name: str) -> str:
    cleaned = "_".join(raw_name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_")


def _build_session_prefix(participant: str) -> str:
    date_prefix = datetime.now().strftime("%m_%d_%y")
    return f"{date_prefix}_{participant}_mental"


def _prompt_session_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        participant = _sanitize_participant_name(raw)
        if participant:
            return _build_session_prefix(participant)
        print("Participant name cannot be empty. Please try again.")


if __name__ == "__main__":
    fname = _prompt_session_prefix()
    print(f"[SESSION] Using filename prefix: {fname}")
    run_task(fname=fname)
