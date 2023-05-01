#!/bin/bash
mkdir ~/.zfunc
poetry completions zsh > ~/.zfunc/_poetry

echo "fpath+=~/.zfunc/_poetry" >> ~/.zshrc
echo "autoload -Uz compinit && compinit" >> ~/.zshrc

pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type pre-merge-commit
