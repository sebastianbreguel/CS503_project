#!/bin/bash

# Check if a message argument is provided
if [ -z "$1" ]; then
  echo "Please provide a commit message."
  exit 1
fi

# Run black formatter
black .

# Run isort
isort .

# Add changes to git
git add .

# Commit changes with the provided message
git commit -m "$1"

# Push changes to the remote repository
git push