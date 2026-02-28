#!/bin/bash

echo "Installing git-lfs..."
git lfs install

echo "Cloning HuggingFace model repo..."
git clone https://huggingface.co/Jssun/aidocent models

echo "Model download complete."