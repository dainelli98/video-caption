# -*- coding: utf-8 -*-
"""Utils - decoder."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("Model-Utils")

df = pd.read_csv("filename")

# Splitting all caption cells into tokens
token_dictionary = np.concatenate(df.iloc[:, 0].str.split().values)

unique_token_dictionary = set(token_dictionary)

print("token_dictionary length: ")
print(len(token_dictionary))
print("unique_token_dictionary length:")
print(len(unique_token_dictionary))
