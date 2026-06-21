# GitHub release checklist

1. Confirm the author name and ORCID in `CITATION.cff`.
2. Replace the placeholder `repository-code` URL.
3. Confirm MIT for code and CC BY 4.0 for repository-authored outputs.
4. Confirm that no source images, personal data, credentials, or restricted metadata are present.
5. After changing metadata, rebuild the file inventory and checksums:

   ```bash
   python tools/rebuild_release_metadata.py
   ```
6. Install Git LFS before adding files:

   ```bash
   git lfs install
   git init
   git add .gitattributes
   git add .
   git commit -m "Release BSPC autofocus benchmark v1.0.0"
   ```

7. Inspect LFS tracking:

   ```bash
   git lfs ls-files
   git status
   ```

8. Create the GitHub repository and push.
9. Connect the repository to Zenodo or upload the prepared Zenodo archive manually.
10. Reserve/publish the Zenodo DOI.
11. Add the final DOI to the GitHub README, `CITATION.cff`, and manuscript data/code availability statement.
12. Create a GitHub release tag matching the archived version, for example `v1.0.0`.
