#!/bin/bash

echo "Installing git-lfs..."
git lfs install

echo "Cloning HuggingFace model repo..."
git clone https://huggingface.co/Jssun/aidocent temp_models

echo "Moving model folders into project..."
mv temp_models/* .

echo "Cleaning up..."
rm -rf temp_models

echo "Model download complete."