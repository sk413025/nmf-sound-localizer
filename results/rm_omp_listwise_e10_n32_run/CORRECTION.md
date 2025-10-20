# Correction for commit bf49ab63cf0f1c3228ed33a6d6cf067bcbf10e92

- The subset_manifest.json in that commit was generated via basename scanning of the data_root.
- Due to duplicate filenames across angle_* folders, all entries resolved to angle_85/* (first match), producing a single-angle manifest.
- That manifest is not representative and must not be used for teacher-vs-GT verification.
- This commit supersedes it with a manifest-only verification workflow where eval loaders write absolute paths + angle_deg, and the verifier requires multi-angle subsets.
