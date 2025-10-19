1. Fries Detection Pipeline
    a. Extend `ui_profiles.json` with calibrated ROIs: `ROI_BUBBLE`, `ROI_SIDES_GRID`, `ROI_SIZE_COL`, `ROI_CONFIRM`. Load them at bot start and provide helpers to convert ROI fractions into absolute pixel rectangles.
    b. When entering phase 2, capture the bubble region, clean it up if needed (denoise, deskew), and surface the crop in the GUI with placeholders for detected type, size, confidence, and the planned click sequence.
    c. Curate fries assets inside `images/sides/`:
        - Type templates: `fries_classic.png`, `fries_thick.png`, `onion_rings.png`.
        - Size templates: `size_S.png`, `size_M.png`, `size_L.png`.
        - Optional manifest (JSON) listing expected scale and color hints to standardize preprocessing.
    d. Implement `detect_fries_request(bubble_img)` using multi-scale template matching plus a simple color mask on the icon background. Return `{type, size, type_conf, size_conf}` and write annotated debug images. Target thresholds: type >= 0.82, size >= 0.80.
    e. Map the detected type/size to grid coordinates via `ui_profiles.json`. If the bubble text contains "no fries", skip straight to the drinks phase.
    f. Execute the UI flow: click the fries type cell, verify the highlight, click the size column, then press confirm. Debounce between clicks (>= 120 ms) and ensure the panel collapses before proceeding.
    g. Add fail-safe logic: if confidence stays below threshold or the flow exceeds 1.3 s overall, press the in-game "Can you repeat?" button, recapture the bubble, and retry with exponential backoff.

2. Drink Detection Pipeline
    a. Add drink templates under `images/drinks/`: `cola.png`, `lemon_lime.png`, `shake.png`. Reuse the size templates from the fries pipeline.
    b. Implement `detect_drink_request(bubble_img)` mirroring the fries detector. Return `{type, size, type_conf, size_conf}` and reuse the shared debug logging helpers.
    c. Write `ensure_drink_panel_visible()`: detect the drink tab icon via template matching; if it is missing, click the tab ROI before issuing any drink clicks.
    d. Display the parsed drink request in the GUI (type, size, confidence, next actions) to simplify debugging and tuning.
    e. If detection yields "no drink" or an empty bubble, transition directly to COMPLETE; otherwise perform drink -> size -> confirm with the same debounce and verification rules as the fries flow.

3. 100% accuracy on ingredient phase
    a. Add patty and veg patty to be identified: add templates `patty_beef.png` and `patty_veg.png` (detect veg via a small leaf overlay near the patty).
    b. Make it so there is no default ingredient: at phase start, reset or clear if available; otherwise only proceed right after a new order bubble is detected.
    c. Detect the amount of each ingredient: in `parse_ingredient_order(bubble_img)`, either count multiple template peaks (with simple NMS) or detect the tiny A-1/A-2/A-3 chips near each icon.
    d. Click each ingredient exactly count times; throttle clicks (<= 85 ms), then only press "Go" when enabled.
    e. Add unit tests on saved bubbles to verify exact tuples like `[('patty_veg', 1), ('cheese', 2), ('lettuce', 1)]`.
    f. Log crops (bubble plus panel) and decisions to `/runs/<date>/session/` to grow the dataset and fix edge cases fast.

4. Build an overall workflow
    a. Add a small state machine with phases: `ORDER_INGREDIENTS -> SIDES_FRIES -> DRINKS -> COMPLETE` (with timeouts and retries).
    b. Phase detection: use OCR anchors ("Hey! I'd like:" = ingredients, "With..." = sides, "And a..." = drinks) or detect which right panel is visible.
    c. Centralize clicks in one helper (`click_named`, `wait_until`) and load all coords/ROIs from `ui_profiles.json` (per resolution).
    d. Add hotkeys (F8 start/stop, F9 pause) and a GUI overlay that shows phase, parsed order, confidence, and next action.
    e. Add acceptance tests: 20 ingredient bubbles, 20 fries requests, 20 drink requests must each reach "Go"; end-to-end average <= 3.5 s/customer at 1080p with >= 95% success.
    f. Keep current files and extend them (no rebuild): `fast_food_bot.py` (loop + GUI), `order_processor.py` (add `parse_ingredient_order`, `detect_fries_request`, `detect_drink_request`), `matcher.py` (add "all peaks" mode plus NMS), `text_finder_orc.py` (phase anchors), JSON configs and `images/` for templates.
