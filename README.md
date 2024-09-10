# Surgical Feedback Detection and Analysis

1. models/ contains the different models we developed
   - ExtractDialogueModel extracts full dialogue interactions between trainer/trainee surgeons in full recordings and 1) detects feedback, 2) classifies feedback components, and 3) assesses feedback effectiveness
   - AudioModel (via. Wav2Vec), TextModel (via. BERT), and AudioTextFusionModel (via. Wav2Vec + BERT) are all binary models that input a 10-sec audio fragment and classify whether it contains feedback.
   - TemporalDetectionModel leverages these 3 binary models and applies them in a rolling window of 10-secs with 5-secs overlap for full recordings of surgery.
2. Training
   - train.py has functions for training the binary models and run_train.py runs them
3. Experiments
   - temporal_detection_*.py contain experiments for temporal detection (our base case for feedback detection)
   - dialogue_exps.py contains experiments for dialogue extraction (our innovation)
4. Other
   - Most other .py are used as imports
   - All .ipynb notebooks are to analyze results and exports
