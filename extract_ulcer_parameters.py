import os
import json
import ollama
import time
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List, Optional
import pandas as pd

# NOTE: PLEASE ENSURE YOU RUN ollama pull llama3.2:1b on terminal before running the script

class UlcerParameters(BaseModel):
    ulcer_centrality: int = Field(..., ge=0, le=9)
    reason_for_ulcer_centrality: str
    ulcer_depth: int = Field(..., ge=0, le=9)
    reason_for_ulcer_depth: str
    corneal_thinning: int = Field(..., ge=0, le=9)
    reason_for_corneal_thinning: str

csv_path = './LLM-3_runs_dataonly_test.csv'  # Update this path to your dataset
processed_data = []
processed_count = 0
start_time = time.time()

prompt_template = """
### Task
You are an expert ophthalmologist and have been tasked with corneal examination from a given patient note. Your task is to identify ulcer parameters (1) Centrality; (2) Depth; (3) Thinning.

### Below are the definitions to determine whether the patient has ulcer parameters:
(1) Ulcer Centrality is described with below descriptors
    a. Ulcer or infiltrate at the center of cornea OR in visual axis OR in front of the pupil OR on the visual axis or central location or total ulcer
    b. Mark as 0 if Ulcer or infiltrate at the periphery of the cornea or mentioned periphery or at 1-12 clock hours or at limbus or superior /inferior/nasal/temporal location or “not in visual axis”
    c. Mark as 9, if both 1 & 2 are not applicable or if details of ulcer or infiltrate location is not available

(2) Ulcer Depth is described with below descriptors
    a. More than or equal to 50% of stromal thickness (>=50%) OR
    b. Deep, full thickness, or dense stromal OR
    c. Upto/ involving posterior stroma or endothelium or endothelial plague
    d. Mark as 0, If mentioned superficial or < 50% deep or <50% depth
    e. Mark as 9, if 1, 2  & 3 are not applicable or if details of ulcer depth is not available

(3) Corneal Thinning is described with below descriptors
    a. More than or equal to 50% of stromal thinning (>=50%) OR
    b. Globe perforation or softness OR
    c. Digital tension is soft or low OR IOP by palpation is soft or low
    d. Eye soft OR
    e. Iris incarceration OR
    f. Note the mention of the Seidel’s positive or Seidel’s negative or its status or Iris incarceration or Impending perforation OR
    g. The note mentions the word ‘thinning’ or ‘descemetocele’ or ‘melt” please consider the cornea is thin.

### Below is the patient's clinical note
{note}

### Below is the format of JSON object with following keys:
1. Ulcer Centrality: int - 1 for Central OR 0 for Not Central OR 9 for absence of ulcer centrality descriptors
2. Reason for Ulcer Centrality: str - your response for presence of ulcer centrality
3. Ulcer Depth: int - 1 for Deep OR 0 for Not Deep OR 9 for absence of ulcer depth descriptors
4. Reason for Ulcer Depth: str - your response for presence of ulcer depth
5. Corneal Thinning: int - 1 for Thinning OR 0 for Not Thinning OR 9 for absence of ulcer centrality descriptors
6. Reason for Corneal Thinning: str - your response for presence of corneal thinning

An example of how your JSON should be formatted is shown below:
json
{{
    "ulcer_centrality": 0/1/9,
    "reason_for_ulcer_centrality": "reason",
    "ulcer_depth": 0/1/9,
    "reason_for_ulcer_depth": "reason",
    "corneal_thinning": 0/1/9,
    "reason_for_corneal_thinning": "reason"
}}
The above example is only for illustration purpose only.
"""

df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    text_content = row['note']  # Ensure the column name is 'note'
    note_id = index + 1

    # Replace {note} with the actual note content
    prompt = prompt_template.format(note=text_content)

    response = ollama.chat(
        model='llama3.2:1b',
        format=UlcerParameters.model_json_schema(),
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )

    try:
        ulcer_data_json = json.loads(response.message.content)
        ulcer_parameters_instance = UlcerParameters(**ulcer_data_json)

        # Create a dictionary for the current note's data
        data_entry = ulcer_parameters_instance.dict()
        data_entry['note_id'] = note_id
        data_entry['note'] = text_content  # Add the 'note' column here
        processed_data.append(data_entry)
        processed_count += 1

        print(f"Processed note ID {note_id} successfully.")
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Failed to process note ID {note_id}: {e}")

total_processing_time = (time.time() - start_time) / 60
print(f"\nTotal processing time for the dataset: {total_processing_time:.2f} minutes.")

processed_df = pd.DataFrame(processed_data)
processed_df.to_csv('extracted_ulcer_parameters.csv', index=False)
