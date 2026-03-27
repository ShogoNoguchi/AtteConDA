# Hugging Face model-card upload notes

Each file in this directory should be uploaded as the `README.md` of the matching Hugging Face model repository.

Mapping:

- `AtteConDA-SDE-Scratch-30K.md` -> `Shogo-Noguchi/AtteConDA-SDE-Scratch-30K`
- `AtteConDA-SDE-UniCon-30K.md` -> `Shogo-Noguchi/AtteConDA-SDE-UniCon-30K`
- `AtteConDA-SDE-UniCon-60K.md` -> `Shogo-Noguchi/AtteConDA-SDE-UniCon-60K`
- `AtteConDA-SDE-UniCon-90K.md` -> `Shogo-Noguchi/AtteConDA-SDE-UniCon-90K`
- `AtteConDA-SDE-UniCon-60K-PAM.md` -> `Shogo-Noguchi/AtteConDA-SDE-UniCon-60K-PAM`
- `AtteConDA-SDE-UniCon-Init.md` -> `Shogo-Noguchi/AtteConDA-SDE-UniCon-Init`

Practical upload method:

1. open the model page in the browser
2. edit the model card / README
3. paste the corresponding file contents
4. save

This is the safest release method because it avoids relying on a CLI syntax that may differ across machines.
