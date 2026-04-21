- modify dockerfile

- check whether the Resizing layer is the correct way of implementing UNet
- consider whether we're sending the (central) model updates through v6, or simply save them in the blob storage and pull from there with the right key (the latter would be better arguably, but also harder to test with mock client - no blob storage setup)
- setup blob storage
- figure out modifications required for using blob storage
- get rid of files that are not required anymore
    - most of the flask stuff I think
- figure out how we're going ot evaluate
