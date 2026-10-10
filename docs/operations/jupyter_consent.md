# Google consent in Jupyter

The notebook uses the existing desktop OAuth client already installed at `/workspace/.secrets/google_client.json`. Generate it with:

```bash
python scripts/create_google_consent_notebook.py --output /workspace/google_drive_consent.ipynb
```

Open the generated notebook in the authenticated Pod Jupyter session, choose **Google Drive consent** (`illegibility_oauth`), and run both code cells. Open the printed Google URL in your browser and select `antoniojun.nyc@gmail.com`. The requested scope is only `drive.file`.

After consent, Google redirects the browser to `http://localhost:8765/`. A connection error is expected because this workflow does not open a server on your computer. Copy the full URL from the address bar and paste it into the notebook's hidden input. The workflow checks the exact callback address and matching state, then exchanges the code over Google's HTTPS endpoint. Use the callback from the current consent attempt.

Do not paste the callback into a cell or chat. No token, client secret, callback or exception text is printed. Actual access-token scopes, refresh authorization and the Drive account are checked before credentials are atomically saved to `/workspace/.secrets/google_token.json` with permissions `0600`; the credentials directory uses `0700`. Set `GOOGLE_OAUTH_TOKEN_FILE` to that path for the experiment. This consent does not itself prove artifact upload or full experiment completion.
