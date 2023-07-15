
# Video Caption


## Team Information
- Team Members
    - Daniel Martín
    - Joan Pascual
    - Juan Bacardit
    - Sergi Taramon
- Advisor: Carlos Escolano
- Framework: Pytorch

## Project Goals
Generate a representative brief description of a small video by: 
- Use transfer learning to encode video information
- Implement transformer decoder to generate captions
    - Extract key frames from a video input using efficient algorithms.
    - Encode these key frames using VideoMae, an advanced video encoder.
    - Combine video embeddings with corresponding captions embeddings.
- Train a transformer decoder to generate captions explaining the video content.
- Achieve valid and accurate captions explaining what happens in a given video input.
- Deploy built solution on Google Cloud

## Environment Setup
- Description of the environment setup

## Dataset
- Selected dataset
- Data preparation

## Methods/Algorithms Used

- Key Frame Extraction: This is the initial step where we extract important frames from the video using an efficient extraction algorithm.
- VideoMae Encoder: We use this advanced encoder to process the extracted key frames into a form that can be used for model training.
- Caption Embeddings: Captions are processed into embeddings that represent the semantic meaning of the text.
- Transformer Decoder: We use a transformer-based model to learn the relationship between the video embeddings and captions embeddings, and generate appropriate captions for new video inputs.

![General data flow](./report/images/general_flow.png)

### Key Frame Extraction:
### Caption Embeddings
### Transformer Decoder

## Model Architecture
- Model design
- Architecture

## Experiments
- Experiment 1
- Experiment 2

## Results Summary
- Summary of results

## Conclusions
- Conclusion 1
- Conclusion 2
- Conclusion 3
- Conclusion 4

## Next Steps
- Next Step 1
- Next Step 2

## References
- Reference 1
- Reference 2
