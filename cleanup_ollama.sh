#!/bin/bash

echo "Current Ollama Models:"
echo "===================="
ollama list
echo ""

read -p "Do you want to delete llama4:scout (67GB)? [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting llama4:scout..."
    ollama rm llama4:scout
    echo ""
    echo "Done! Remaining models:"
    ollama list
    echo ""
    echo "Freed up ~67GB of disk space!"
else
    echo "Cancelled. No models deleted."
fi
