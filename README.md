# ASL Glove — Real-Time Sensing and ML

A wearable glove system for real-time American Sign Language (ASL) gesture recognition using multi-sensor hand motion sensing and machine learning.

## Motivation
High-quality ASL interpretation requires accurate capture of fine-grained hand and finger motion.
This project explores a sensing-to-inference pipeline that enables real-time gesture decoding with minimal latency.

## System Overview
The system implements an end-to-end pipeline:
1. Hand and finger motion sensing
2. Signal filtering and alignment
3. Feature extraction
4. Real-time ML-based gesture inference
5. Client–server communication for streaming and decoding

## Implementation
- Real-time data streaming via a client–receiver architecture
- Signal preprocessing and filtering for robustness
- Modular processing and inference components designed for low latency

## Code Structure
- `client.py`  
  Sensor data streaming client
- `realtime_model_receiver.py`  
  Receives streamed data and manages real-time inference
- `realtime_model_processor.py`  
  Model processing and gesture decoding
- `filterforeverything.py`  
  Signal filtering and preprocessing utilities
- `single_add.py`  
  Supporting utility functions

## Usage
This repository provides source code for the ASL glove system.
Data files and trained models are excluded.

