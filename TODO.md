# TODO: Fix API Fetch Error in Soccer Predictor App

## Steps to Complete:

1. **Update config.py**:
   - Change SEASON from 2023 to 2024 to target current season data.

2. **Modify data_fetcher.py**:
   - In `create_training_dataset()`, add a fallback to generate mock sample data (50 fictional matches with realistic stats) if the API fetch returns no fixtures. This allows the app to work for demo/training without API dependency.
   - Ensure mock data includes features like shots, possession, outcome, etc., matching the extract_match_features() structure.

3. **Test the Changes**:
   - Restart the Streamlit app: `streamlit run app.py`.
   - Click "Fetch Match Data" in sidebar – should succeed with mock data if API fails.
   - Verify data overview displays, model training works, and predictions are possible.

4. **Optional Cleanup**:
   - Once real API works (e.g., after subscription upgrade), remove mock fallback code.

Progress: [x] Step 1 [x] Step 2 [ ] Step 3 [ ] Step 4
