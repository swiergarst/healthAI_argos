# healthAI - Argos usecase working document

I started this while most of the implementation has been finished, but up until now, the work has been relatively straightforward, mostly things like updating code to be compatible with newer version of tensorflow and such. starting this document as I'm running into things/changes that might require further thinking / discussing


## Main issues to discuss with people / test

- current implementation uses one batch per iteration, thereby requiring no weighted averaging. I'd assume you go over all of your batches per round (and thereby also need weighted averaging). should we implement it like that?

- changing s.t. we send over model updates back to the central part of the task. I believe this should now be safe, as these updates are encrypted (and stored as encrypted in the blob storage), and can therefore only be accessed from the central 'task', not from the researcher itself directly

## Diary

### 23rd of April
- training workflow was already tested before, now starting to add the federated part to it, so implementing the central task essentially
- Ananya's averaging implementation looks weird and I don't like it so I'm gonna reimplement it.
- question is whether we need weighted averaging or not, with the current workflow it seems not needed (but maybe that workflow should change)
- turned out that ananya's implementations was quite efficient with how tensorflow does model weights so I'm keeping it for now.


### 28th of April
